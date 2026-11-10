"""服务入口：python -m app.server"""
from __future__ import annotations

import argparse

import uvicorn

from . import capture, config, db, trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 训练平台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8750)
    parser.add_argument("--public", action="store_true", help="监听 0.0.0.0，允许局域网访问")
    args = parser.parse_args()

    config.ensure_dirs()
    db.query("SELECT 1")            # 初始化 DB
    trainer.recover_orphans()        # 标记上次遗留的运行中任务
    capture.recover()                # 采集中断标记
    capture.start_scheduler()        # 屏幕采集调度器

    host = "0.0.0.0" if args.public else args.host
    print("=" * 56)
    print("  YOLO 训练平台")
    print(f"  http://{'127.0.0.1' if args.public else host}:{args.port}")
    print("=" * 56)
    uvicorn.run("app.main:app", host=host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
