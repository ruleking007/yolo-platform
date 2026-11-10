"""屏幕采集插件：后台定时任务，用户移动鼠标时每隔 N 秒截屏入库。

设计：
  - 调度器线程每秒扫描到期任务（定时一次 / 每天定点），到点自动开跑
  - 每个任务独立线程：检测鼠标位移判定"人在用电脑"，活跃才截屏
  - 截屏经 ingest_images 规范化入库：自动与数据集已有图片去重（感知哈希）
  - 磁盘余量、连续失败、最大张数、时长上限都有保护，异常自动收尾
"""
from __future__ import annotations

import ctypes
import datetime as dt
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

from . import config, datasets, db

_IS_WIN = sys.platform.startswith("win")

_lock = threading.Lock()
_threads: dict[str, tuple[threading.Thread, threading.Event]] = {}
_sched_started = False


# ---------------------------------------------------------------- 光标活动检测

class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def cursor_pos() -> tuple[int, int] | None:
    if not _IS_WIN:
        return None
    try:
        p = _POINT()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(p)):
            return (p.x, p.y)
    except Exception:
        pass
    return None


def screen_size() -> tuple[int, int]:
    if _IS_WIN:
        try:
            u = ctypes.windll.user32
            return (u.GetSystemMetrics(0), u.GetSystemMetrics(1))
        except Exception:
            pass
    return (0, 0)


def grab_screen(dest_dir: Path | None = None) -> Path | None:
    """截主屏保存为 JPEG 临时文件，返回路径（调用方负责删除）。"""
    from PIL import ImageGrab
    img = ImageGrab.grab()
    if img is None:
        return None
    dest_dir = dest_dir or config.TMP_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / f"cap_{int(time.time() * 1000)}.jpg"
    img.convert("RGB").save(tmp, "JPEG", quality=88)
    return tmp


def capture_available() -> tuple[bool, str]:
    try:
        from PIL import ImageGrab  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"缺少截屏能力：{e}"
    if _IS_WIN:
        return True, "ok"
    return True, "ok"  # 非 Windows 尽力而为


# ---------------------------------------------------------------- 调度

def _next_daily(hhmm: str) -> float:
    h, m = (int(x) for x in hhmm.split(":")[:2])
    now = dt.datetime.now()
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if target.timestamp() <= time.time():
        target += dt.timedelta(days=1)
    return target.timestamp()


def _once_epoch(at_iso: str) -> float:
    return dt.datetime.fromisoformat(at_iso).timestamp()


def compute_next_run(job: dict) -> float | None:
    try:
        if job["mode"] == "daily":
            return _next_daily(job["at_time"])
        if job["mode"] == "once":
            return _once_epoch(job["at_time"])
    except Exception:
        return None
    return None


def start_scheduler() -> None:
    global _sched_started
    if _sched_started:
        return
    _sched_started = True
    threading.Thread(target=_sched_loop, daemon=True, name="capture-scheduler").start()


def _sched_loop() -> None:
    while True:
        try:
            for job in db.capture_due(time.time()):
                start_job(job["id"])
        except Exception:
            pass
        time.sleep(1.0)


def recover() -> None:
    """服务重启后：把 running 的采集任务标记为 interrupted。"""
    for j in db.list_capture_jobs():
        if j["status"] == "running":
            db.update_capture_job(j["id"], status="interrupted",
                                  last_error="服务重启导致采集中断")


def start_job(job_id: str) -> bool:
    with _lock:
        t = _threads.get(job_id)
        if t and t[0].is_alive():
            return False
        job = db.get_capture_job(job_id)
        if not job:
            return False
        stop = threading.Event()
        th = threading.Thread(target=_run_loop, args=(job_id, stop),
                              daemon=True, name=f"capture-{job_id}")
        _threads[job_id] = (th, stop)
        db.update_capture_job(job_id, status="running", started_at=db.now(),
                              last_error=None)
        th.start()
        return True


def stop_job(job_id: str) -> bool:
    with _lock:
        t = _threads.get(job_id)
    if t and t[0].is_alive():
        t[1].set()
        return True
    job = db.get_capture_job(job_id)
    if job and job["status"] in ("scheduled", "running", "interrupted"):
        db.update_capture_job(job_id, status="paused")
        return True
    return False


def reschedule(job_id: str) -> bool:
    """重新排程（暂停/完成后恢复）。"""
    job = db.get_capture_job(job_id)
    if not job:
        return False
    nxt = compute_next_run(job)
    db.update_capture_job(job_id, status="scheduled", next_run=nxt)
    return True


# ---------------------------------------------------------------- 采集主循环

_MIN_INTERVAL = 0.5


def _run_loop(job_id: str, stop: threading.Event) -> None:
    job = db.get_capture_job(job_id)
    if not job:
        return
    ds_id = job["dataset_id"]
    interval = max(_MIN_INTERVAL, float(job.get("interval_sec") or 1))
    window = max(1.0, float(job.get("active_window") or 5))
    threshold = max(1, int(job.get("move_threshold") or 3))
    max_images = int(job.get("max_images") or 600)
    dur_min = float(job.get("duration_min") or 10)
    deadline = time.time() + dur_min * 60

    tmpdir = config.TMP_DIR / f"cap_{job_id}"
    tmpdir.mkdir(parents=True, exist_ok=True)

    captured = int(job.get("captured") or 0)
    dup = int(job.get("skipped_dup") or 0)
    idle = int(job.get("skipped_idle") or 0)
    fails = 0
    err = ""
    prev = cursor_pos()
    last_move = 0.0

    try:
        while not stop.is_set() and time.time() < deadline and captured < max_images:
            # 磁盘保护
            if shutil.disk_usage(config.DATA_DIR).free < 1 << 30:
                err = "磁盘剩余不足 1GB，任务自动停止"
                break
            now = time.time()
            pos = cursor_pos()
            if pos is None:                      # 无法检测（非 Windows）→ 视为活跃
                active = True
            else:
                if prev is not None and (abs(pos[0] - prev[0]) + abs(pos[1] - prev[1])) >= threshold:
                    last_move = now
                prev = pos
                active = (now - last_move) <= window

            if not active:
                idle += 1
                db.update_capture_job(job_id, skipped_idle=idle)
                _sleep_interval(stop, interval)
                continue

            try:
                fp = grab_screen()
                if fp is None:
                    raise RuntimeError("截屏返回空")
                res = datasets.ingest_images(ds_id, [(fp.name, fp)],
                                             split="train", max_side=1920, dedupe=True)
                fp.unlink(missing_ok=True)
                captured += res["added"]
                dup += res["skipped_duplicate"]
                fails = 0
                db.update_capture_job(job_id, captured=captured, skipped_dup=dup,
                                      last_capture=db.now())
            except Exception as e:  # noqa: BLE001
                fails += 1
                if fails >= 10:
                    err = f"连续 10 次截屏失败：{e}"
                    break
            _sleep_interval(stop, interval)

        # 收尾状态
        if stop.is_set():
            db.update_capture_job(job_id, status="paused", finished_at=db.now())
        elif err:
            db.update_capture_job(job_id, status="failed", finished_at=db.now(),
                                  last_error=err)
        elif job.get("mode") == "daily":
            db.update_capture_job(job_id, status="scheduled", finished_at=db.now(),
                                  next_run=_next_daily(job["at_time"]))
        else:
            db.update_capture_job(job_id, status="done", finished_at=db.now())
    except Exception as e:  # noqa: BLE001
        db.update_capture_job(job_id, status="failed", finished_at=db.now(),
                              last_error=f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        with _lock:
            _threads.pop(job_id, None)


def _sleep_interval(stop: threading.Event, seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end and not stop.is_set():
        time.sleep(0.15)


# ---------------------------------------------------------------- 辅助

def create_job(dataset_id: str, name: str, mode: str, at: str,
               duration_min: float, interval_sec: float, active_window: float,
               move_threshold: int, max_images: int) -> dict:
    if mode not in ("once", "daily"):
        raise ValueError("mode 只能是 once 或 daily")
    if not at:
        raise ValueError("请填写启动时间")
    job = {
        "id": db.new_id("cap_"),
        "name": name or "屏幕采集",
        "dataset_id": dataset_id,
        "mode": mode,
        "at_time": at,
        "duration_min": max(0.5, min(float(duration_min), 480)),
        "interval_sec": max(_MIN_INTERVAL, float(interval_sec)),
        "active_window": max(1.0, float(active_window)),
        "move_threshold": max(1, int(move_threshold)),
        "max_images": max(1, int(max_images)),
    }
    nxt = compute_next_run(job)
    if nxt is None:
        raise ValueError("时间格式不正确")
    db.create_capture_job(job, nxt)
    return db.get_capture_job(job["id"])


def recent_captures(dataset_id: str, limit: int = 12) -> list[dict]:
    root = datasets.images_dir(dataset_id)
    items: list[tuple[float, str, str]] = []
    for split in ("train", "val"):
        d = root / split
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.name.startswith("cap_") and f.suffix.lower() in config.IMAGE_EXTS:
                items.append((f.stat().st_mtime, split, f.name))
    items.sort(reverse=True)
    return [{"mtime": t, "split": s, "name": n} for t, s, n in items[:limit]]


def job_status(job: dict) -> dict:
    """给前端的友好状态：倒计时、是否正在运行。"""
    out = dict(job)
    with _lock:
        t = _threads.get(job["id"])
    out["thread_alive"] = bool(t and t[0].is_alive())
    if job["status"] == "scheduled" and job.get("next_run"):
        out["secs_to_run"] = max(0, round(job["next_run"] - time.time()))
    return out
