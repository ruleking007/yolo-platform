"""训练任务编排：启动/停止子进程、读取进度与日志、清理僵尸任务。"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import config, db

_PROCS: dict[str, subprocess.Popen] = {}
_IS_WIN = sys.platform.startswith("win")


def _log_path(task_id: str) -> str:
    return str(config.RUNS_DIR / task_id / "train.log")


def _creation_flags() -> int:
    if not _IS_WIN:
        return 0
    return subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]


def _child_env() -> dict:
    return {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}


def start(task_id: str) -> None:
    """以独立子进程启动训练，输出重定向到 train.log。"""
    task = db.get_task(task_id)
    if not task:
        raise ValueError("任务不存在")
    log = Path(_log_path(task_id))
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "ab", buffering=0)
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "app.train_worker", task_id],
            cwd=str(config.BASE_DIR),
            stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            creationflags=_creation_flags(), close_fds=True, env=_child_env(),
        )
    finally:
        fh.close()
    _PROCS[task_id] = proc
    db.update_task(task_id, pid=proc.pid)


def _kill_pid(pid: int) -> None:
    if _IS_WIN:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, check=False)
    else:
        import signal as _sig
        try:
            os.killpg(os.getpgid(pid), _sig.SIGTERM)
        except Exception:
            os.kill(pid, _sig.SIGTERM)


def stop(task_id: str) -> bool:
    """终止训练进程。"""
    task = db.get_task(task_id)
    if not task:
        return False
    proc = _PROCS.pop(task_id, None)
    killed = False
    if proc is not None and proc.poll() is None:
        try:
            proc.kill()
            proc.wait(timeout=8)
            killed = True
        except Exception:
            killed = True
    if not killed and task.get("pid"):
        try:
            _kill_pid(int(task["pid"]))
            killed = True
        except Exception:
            pass
    db.update_task(task_id, status="stopped", finished_at=db.now(), pid=None)
    if task.get("run_dir"):
        try:
            from .train_worker import write_state
            write_state(Path(task["run_dir"]), status="stopped", phase="已手动停止")
        except Exception:
            pass
    return killed


def is_alive(task_id: str) -> bool:
    proc = _PROCS.get(task_id)
    if proc is not None:
        return proc.poll() is None
    task = db.get_task(task_id)
    pid = (task or {}).get("pid")
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except Exception:
        try:
            if _IS_WIN:
                out = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {int(pid)}"],
                    capture_output=True, text=True, check=False).stdout
                return str(pid) in out
            os.kill(int(pid), 0)
            return True
        except Exception:
            return False


def tail_log(task_id: str, offset: int = 0, max_bytes: int = 80000) -> dict:
    """按字节偏移增量读取日志。"""
    task = db.get_task(task_id)
    if not task:
        return {"offset": 0, "text": "", "done": True}
    p = Path(_log_path(task_id))
    running = task["status"] in ("queued", "running")
    if not p.exists():
        return {"offset": 0, "text": "", "done": not running}
    size = p.stat().st_size
    if offset > size or offset < 0:
        offset = 0
    with open(p, "rb") as fh:
        fh.seek(offset)
        data = fh.read(max_bytes)
    return {"offset": offset + len(data), "text": data.decode("utf-8", errors="replace"),
            "done": not running}


def read_state(task_id: str) -> dict:
    task = db.get_task(task_id)
    if not task or not task.get("run_dir"):
        return {}
    sp = Path(task["run_dir"]) / "platform_state.json"
    if not sp.exists():
        return {}
    try:
        return json.loads(sp.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_metrics(task_id: str) -> list[dict]:
    task = db.get_task(task_id)
    if not task or not task.get("run_dir"):
        return []
    mp = Path(task["run_dir"]) / "platform_metrics.jsonl"
    if not mp.exists():
        return []
    out = []
    for line in mp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def progress_of(task: dict) -> float:
    if task.get("status") == "completed":
        return 1.0
    state = read_state(task["id"])
    ep, tot = state.get("epoch"), state.get("total_epochs")
    if ep and tot:
        return min(0.99, round(float(ep) / float(tot), 4))
    return float(task.get("progress") or 0.0)


def recover_orphans() -> None:
    """服务启动时：把 DB 里 running/queued 但进程已消失的任务标记为 interrupted。"""
    for task in db.running_tasks():
        if not is_alive(task["id"]):
            db.update_task(task["id"], status="interrupted", finished_at=db.now(),
                           error="训练进程已退出（可能因服务重启），可从 last.pt 续训")


def delete(task_id: str) -> None:
    task = db.get_task(task_id)
    if not task:
        return
    if task["status"] in ("queued", "running"):
        stop(task_id)
    shutil.rmtree(config.RUNS_DIR / task_id, ignore_errors=True)
    db.delete_task(task_id)
