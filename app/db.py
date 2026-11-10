"""SQLite 元数据存储：训练任务与模型仓库。"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Any

from . import config

_LOCK = threading.RLock()
_CONNECTION: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    dataset_id   TEXT,
    model        TEXT,
    status       TEXT NOT NULL,          -- queued|running|completed|failed|stopped|interrupted
    args         TEXT,                   -- JSON
    log_path     TEXT,
    run_dir      TEXT,
    pid          INTEGER,
    progress     REAL DEFAULT 0,
    metrics      TEXT,                   -- JSON（最新一轮指标）
    best_model   TEXT,
    error        TEXT,
    created_at   REAL,
    started_at   REAL,
    finished_at  REAL
);
CREATE TABLE IF NOT EXISTS models (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    task_id      TEXT,
    dataset_id   TEXT,
    base_model   TEXT,
    path         TEXT,
    classes      TEXT,                   -- JSON list
    nc           INTEGER,
    metrics      TEXT,                   -- JSON
    size         INTEGER,
    source       TEXT,                   -- trained|uploaded|demo|pretrained
    task         TEXT,                   -- detect|segment|classify|pose|obb
    note         TEXT,
    created_at   REAL
);
CREATE TABLE IF NOT EXISTS annotations_meta (
    dataset_id   TEXT NOT NULL,
    image        TEXT NOT NULL,
    updated_at   REAL,
    PRIMARY KEY (dataset_id, image)
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS capture_jobs (
    id             TEXT PRIMARY KEY,
    name           TEXT,
    dataset_id     TEXT NOT NULL,
    mode           TEXT NOT NULL,          -- once | daily
    at_time        TEXT NOT NULL,          -- once: ISO 日期时间; daily: HH:MM
    duration_min   REAL,
    interval_sec   REAL,
    active_window  REAL,
    move_threshold INTEGER,
    max_images     INTEGER,
    status         TEXT,                   -- scheduled|running|paused|done|failed|interrupted
    next_run       REAL,
    created_at     REAL,
    started_at     REAL,
    finished_at    REAL,
    last_capture   REAL,
    captured       INTEGER DEFAULT 0,
    skipped_dup    INTEGER DEFAULT 0,
    skipped_idle   INTEGER DEFAULT 0,
    last_error     TEXT
);
"""


def _conn() -> sqlite3.Connection:
    global _CONNECTION
    if _CONNECTION is None:
        _CONNECTION = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
        _CONNECTION.row_factory = sqlite3.Row
        _CONNECTION.execute("PRAGMA journal_mode=WAL")
        _CONNECTION.executescript(SCHEMA)
        _migrate(_CONNECTION)
        _CONNECTION.commit()
    return _CONNECTION


def _migrate(conn: sqlite3.Connection) -> None:
    """为已有库补齐新增列（SQLite 不支持 ADD COLUMN IF NOT EXISTS）。"""
    want = {
        "models": {"task": "TEXT", "note": "TEXT"},
        "tasks": {},
    }
    for table, cols in want.items():
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for col, typ in cols.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")


def query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    with _LOCK:
        cur = _conn().execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def query_one(sql: str, params: tuple = ()) -> dict[str, Any] | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: tuple = ()) -> None:
    with _LOCK:
        c = _conn()
        c.execute(sql, params)
        c.commit()


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def now() -> float:
    return time.time()


# ---------------- tasks ----------------

def create_task(name: str, dataset_id: str, model: str, args: dict, log_path: str) -> str:
    tid = new_id("t_")
    execute(
        "INSERT INTO tasks (id,name,dataset_id,model,status,args,log_path,progress,created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (tid, name, dataset_id, model, "queued", json.dumps(args, ensure_ascii=False), log_path, 0.0, now()),
    )
    return tid


def get_task(task_id: str) -> dict | None:
    return query_one("SELECT * FROM tasks WHERE id=?", (task_id,))


def list_tasks(limit: int = 200) -> list[dict]:
    rows = query("SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,))
    for r in rows:
        r["args"] = json.loads(r["args"] or "{}")
        r["metrics"] = json.loads(r["metrics"] or "{}")
    return rows


def update_task(task_id: str, **fields: Any) -> None:
    if "args" in fields and isinstance(fields["args"], (dict, list)):
        fields["args"] = json.dumps(fields["args"], ensure_ascii=False)
    if "metrics" in fields and isinstance(fields["metrics"], (dict, list)):
        fields["metrics"] = json.dumps(fields["metrics"], ensure_ascii=False)
    if not fields:
        return
    keys = list(fields)
    sql = "UPDATE tasks SET " + ",".join(f"{k}=?" for k in keys) + " WHERE id=?"
    execute(sql, tuple(fields[k] for k in keys) + (task_id,))


def delete_task(task_id: str) -> None:
    execute("DELETE FROM tasks WHERE id=?", (task_id,))


def running_tasks() -> list[dict]:
    return query("SELECT * FROM tasks WHERE status IN ('queued','running')")


# ---------------- models ----------------

def add_model(name: str, path: str, classes: list[str], metrics: dict | None = None,
              task_id: str | None = None, dataset_id: str | None = None,
              base_model: str | None = None, source: str = "trained", size: int = 0,
              task: str = "detect", note: str | None = None) -> str:
    mid = new_id("m_")
    execute(
        "INSERT INTO models (id,name,task_id,dataset_id,base_model,path,classes,nc,metrics,size,source,task,note,created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (mid, name, task_id, dataset_id, base_model, path, json.dumps(classes, ensure_ascii=False),
         len(classes), json.dumps(metrics or {}, ensure_ascii=False), size, source, task, note, now()),
    )
    return mid


def list_models() -> list[dict]:
    rows = query("SELECT * FROM models ORDER BY created_at DESC")
    for r in rows:
        r["classes"] = json.loads(r["classes"] or "[]")
        r["metrics"] = json.loads(r["metrics"] or "{}")
    return rows


def get_model(model_id: str) -> dict | None:
    row = query_one("SELECT * FROM models WHERE id=?", (model_id,))
    if row:
        row["classes"] = json.loads(row["classes"] or "[]")
        row["metrics"] = json.loads(row["metrics"] or "{}")
    return row


def delete_model(model_id: str) -> None:
    execute("DELETE FROM models WHERE id=?", (model_id,))


def touch_annotation(dataset_id: str, image: str) -> None:
    execute(
        "INSERT INTO annotations_meta (dataset_id,image,updated_at) VALUES (?,?,?) "
        "ON CONFLICT(dataset_id,image) DO UPDATE SET updated_at=excluded.updated_at",
        (dataset_id, image, now()),
    )


# ---------------- capture_jobs ----------------

_CAPTURE_COLS = ("id,name,dataset_id,mode,at_time,duration_min,interval_sec,active_window,"
                 "move_threshold,max_images,status,next_run,created_at,started_at,finished_at,"
                 "last_capture,captured,skipped_dup,skipped_idle,last_error")


def create_capture_job(job: dict, next_run: float) -> None:
    execute(
        "INSERT INTO capture_jobs (id,name,dataset_id,mode,at_time,duration_min,interval_sec,"
        "active_window,move_threshold,max_images,status,next_run,created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (job["id"], job["name"], job["dataset_id"], job["mode"], job["at_time"],
         job["duration_min"], job["interval_sec"], job["active_window"],
         job["move_threshold"], job["max_images"], "scheduled", next_run, now()),
    )


def get_capture_job(job_id: str) -> dict | None:
    return query_one("SELECT * FROM capture_jobs WHERE id=?", (job_id,))


def list_capture_jobs(limit: int = 100) -> list[dict]:
    return query("SELECT * FROM capture_jobs ORDER BY created_at DESC LIMIT ?", (limit,))


def capture_due(ts: float) -> list[dict]:
    return query("SELECT * FROM capture_jobs WHERE status='scheduled' AND next_run<=?",
                 (ts,))


def update_capture_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    keys = list(fields)
    execute("UPDATE capture_jobs SET " + ",".join(f"{k}=?" for k in keys) + " WHERE id=?",
            tuple(fields[k] for k in keys) + (job_id,))


def delete_capture_job(job_id: str) -> None:
    execute("DELETE FROM capture_jobs WHERE id=?", (job_id,))


# ---------------- settings ----------------

def get_setting(key: str, default: str | None = None) -> str | None:
    row = query_one("SELECT value FROM settings WHERE key=?", (key,))
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    execute("INSERT INTO settings (key,value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
