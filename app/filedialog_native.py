"""系统原生「打开文件」对话框：让用户用资源管理器挑选本机文件，拿到绝对路径。

浏览器出于安全沙箱限制，网页里的 <input type=file> 永远拿不到真实完整路径
（只有文件名 + 内容）。要拿到路径必须由服务端弹出系统对话框。

注意：
- 必须在独立线程运行（Tk 事件循环会阻塞，且需自己的 mainloop）
- 服务若在无交互桌面（如 SSH、服务账户、计划任务）下运行，会失败；
  调用方应捕获异常并降级为「手填路径 / 网页上传」方案
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

WEIGHT_FILTERS: list[tuple[str, str]] = [
    ("模型权重文件", "*.pt *.pth *.onnx *.engine *.tflite *.mlmodel"),
    ("PyTorch 权重", "*.pt *.pth"),
    ("ONNX 模型", "*.onnx"),
    ("TensorRT 引擎", "*.engine"),
    ("所有文件", "*.*"),
]

IMAGE_FILTERS: list[tuple[str, str]] = [
    ("图片文件", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff *.heic *.heif"),
    ("所有文件", "*.*"),
]

ANY_FILTERS: list[tuple[str, str]] = [("所有文件", "*.*")]

_DEFAULT_TIMEOUT = 300  # 用户浏览/选择可能需要时间


class _DialogHandle:
    """持有正在运行的 Tk 根窗口，超时时可从外部强制关闭，避免留下僵尸窗口。"""

    def __init__(self) -> None:
        self.root = None
        self.lock = threading.Lock()
        self.closed = False

    def set_root(self, root: Any) -> None:
        with self.lock:
            self.root = root

    def clear(self) -> None:
        """清掉对 Tk 窗口的引用，避免别的线程 GC 到它（会触发 Tcl_AsyncDelete）。"""
        with self.lock:
            self.root = None

    def force_close(self) -> None:
        with self.lock:
            self.closed = True
            root = self.root
        if root is not None:
            try:
                root.quit()      # 退出 mainloop
            except Exception:
                pass
            try:
                root.destroy()
            except Exception:
                pass


def available() -> tuple[bool, str]:
    """判断当前环境能否弹出原生对话框（返回 可用?, 原因）。"""
    if not sys.platform.startswith("win") and sys.platform != "darwin" and not sys.platform.startswith("linux"):
        return False, f"不支持的平台：{sys.platform}"
    try:
        import tkinter  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"缺少 tkinter：{e}"
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.destroy()
    except Exception as e:  # noqa: BLE001
        return False, f"没有可用的图形桌面会话：{e}"
    return True, "ok"


def _find_dialog_windows(title: str) -> list[int]:
    """在当前桌面枚举可见窗口，按标题找对话框句柄（Windows 专用）。"""
    if not sys.platform.startswith("win"):
        return []
    try:
        import ctypes
        import ctypes.wintypes as w

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        found: list[int] = []

        def cb(hwnd, lp):  # noqa: ARG001
            if user32.IsWindowVisible(hwnd):
                n = user32.GetWindowTextLengthW(hwnd)
                if n:
                    buf = ctypes.create_unicode_buffer(n + 1)
                    user32.GetWindowTextW(hwnd, buf, n + 1)
                    if buf.value.strip() and title in buf.value:
                        found.append(int(hwnd))
            return True

        CB = ctypes.WINFUNCTYPE(ctypes.c_bool, w.HWND, w.LPARAM)
        user32.EnumWindows(CB(cb), 0)
        return found
    except Exception:
        return []


def _close_by_title(title: str) -> bool:
    """给标题匹配的窗口发 WM_CLOSE。模态文件对话框会响应并退出。"""
    if not sys.platform.startswith("win"):
        return False
    hwnds = _find_dialog_windows(title)
    if not hwnds:
        return False
    try:
        import ctypes
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        WM_CLOSE = 0x0010
        for h in hwnds:
            user32.PostMessageW(h, WM_CLOSE, 0, 0)
        return True
    except Exception:
        return False


def _run_dialog(kind: str, filters: list[tuple[str, str]],
                title: str, initialdir: str | None,
                multiple: bool, out: queue.Queue,
                handle: _DialogHandle, deadline: float) -> None:
    """在独立线程里跑 Tk 对话框（每次新建独立 Tk 解释器，用完销毁）。"""
    root = None
    ticker_done = {"v": False}

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        handle.set_root(root)
        root.withdraw()
        root.attributes("-topmost", True)          # 置顶，避免被浏览器挡住
        root.update()

        def watchdog() -> None:
            """超时则向对话框窗口发 WM_CLOSE，让它优雅退出。"""
            if ticker_done["v"]:
                return
            if time.time() >= deadline:
                ticker_done["v"] = True
                if not _close_by_title(title):
                    try:
                        root.quit()
                    except Exception:
                        pass
                return
            try:
                root.after(300, watchdog)
            except Exception:
                pass

        root.after(300, watchdog)

        kwargs: dict[str, Any] = {"title": title, "parent": root}
        if initialdir and Path(initialdir).is_dir():
            kwargs["initialdir"] = str(initialdir)

        if kind == "dir":
            picked = filedialog.askdirectory(**kwargs)
            out.put({"paths": [picked] if picked else [], "ok": bool(picked)})
        elif kind == "save":
            kwargs["defaultextension"] = ".json"
            picked = filedialog.asksaveasfilename(**kwargs)
            out.put({"paths": [picked] if picked else [], "ok": bool(picked)})
        else:
            kwargs["filetypes"] = filters
            if multiple:
                picked = filedialog.askopenfilenames(**kwargs)
                paths = list(picked) if picked else []
            else:
                picked = filedialog.askopenfilename(**kwargs)
                paths = [picked] if picked else []
            out.put({"paths": paths, "ok": bool(paths)})
    except Exception as e:  # noqa: BLE001
        out.put({"error": f"{type(e).__name__}: {e}", "ok": False, "paths": []})
    finally:
        ticker_done["v"] = True
        handle.clear()
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
            # Tcl 解释器必须在创建它的线程里销毁，且不能跨线程被 GC
            del root


def pick_files(filters: list[tuple[str, str]] | None = None,
               title: str = "选择文件",
               initialdir: str | None = None,
               multiple: bool = False,
               timeout: float = _DEFAULT_TIMEOUT,
               kind: str = "file") -> dict[str, Any]:
    """弹出系统对话框，返回 {ok, paths, cancelled, error}。

    阻塞直到用户选择、取消，或超时。
    """
    if kind not in ("file", "dir", "save"):
        return {"ok": False, "paths": [], "error": f"未知类型：{kind}"}

    filt = filters or WEIGHT_FILTERS
    ok, reason = available()
    if not ok:
        return {"ok": False, "paths": [], "error": reason, "unsupported": True}

    q: queue.Queue = queue.Queue()
    handle = _DialogHandle()
    deadline = time.time() + timeout
    t = threading.Thread(target=_run_dialog,
                         args=(kind, filt, title, initialdir, multiple, q, handle, deadline),
                         daemon=True)
    t.start()

    try:
        result = q.get(timeout=timeout + 3)   # 宽限 3 秒等 watchdog 收尾
    except queue.Empty:
        return {"ok": False, "paths": [], "cancelled": True,
                "error": "对话框超时未响应"}

    if result.get("error"):
        return {"ok": False, "paths": [], "error": result["error"]}
    if not result.get("ok"):
        return {"ok": False, "paths": [], "cancelled": True}
    return {"ok": True, "paths": result["paths"], "cancelled": False}


def pick_weight_file(initialdir: str | None = None, multiple: bool = False) -> dict[str, Any]:
    return pick_files(WEIGHT_FILTERS, "选择模型权重文件", initialdir, multiple)


def pick_image_files(initialdir: str | None = None, multiple: bool = True) -> dict[str, Any]:
    return pick_files(IMAGE_FILTERS, "选择图片文件", initialdir, multiple)
