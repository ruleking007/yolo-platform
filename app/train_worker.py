"""训练子进程：由 trainer.py 以 `python -m app.train_worker <task_id>` 启动。

职责：
  1. 从 DB 读取任务参数
  2. 调用 ultralytics 训练，注册回调把逐轮指标写入 run_dir/platform_metrics.jsonl
  3. 维护 run_dir/platform_state.json（epoch 进度 / 阶段），供 Web 轮询
  4. 训练结束后把 best.pt 注册进模型仓库
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

_IS_WIN = sys.platform.startswith("win")


def _state_path(run_dir: Path) -> Path:
    return run_dir / "platform_state.json"


def _metrics_path(run_dir: Path) -> Path:
    return run_dir / "platform_metrics.jsonl"


def write_state(run_dir: Path, **kw) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    p = _state_path(run_dir)
    state = {}
    if p.exists():
        try:
            state = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    state.update(kw)
    state["updated_at"] = time.time()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def append_metric(run_dir: Path, metric: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(_metrics_path(run_dir), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(metric, ensure_ascii=False) + "\n")


def _ensure_fonts(classes: list[str]) -> None:
    """类名含中文等非 ASCII 时，确保 ultralytics 配置目录里有 Unicode 字体。

    ultralytics 画图前会联网下载 Arial.Unicode.ttf（22MB），国内速度极慢会把
    训练卡死在准备阶段。这里直接从本机 Windows 字体目录复制，一次到位。
    """
    import shutil

    if all(all(ord(ch) < 128 for ch in str(n)) for n in classes):
        return                                   # 纯 ASCII 类名不需要
    try:
        from ultralytics.utils import USER_CONFIG_DIR
    except Exception:
        return
    target = USER_CONFIG_DIR / "Arial.Unicode.ttf"
    if target.exists() and target.stat().st_size > 1_000_000:
        return
    candidates = [
        Path("C:/Windows/Fonts/ARIALUNI.ttf"),    # 字体名即 Arial Unicode MS，首选
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for c in candidates:
        if c.exists():
            try:
                USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(c, target)
                print(f"[worker] 中文字体已就位（复制自 {c}）", flush=True)
                return
            except Exception:
                continue


def main(task_id: str) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import config, db, datasets, inference

    def inference_ensure(w: str) -> str:
        return inference.ensure_pretrained(w.replace(".pt", ""))

    task = db.get_task(task_id)
    if not task:
        print(f"[worker] 任务不存在: {task_id}", flush=True)
        return 2

    raw_args = task["args"]
    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
    ds_id = task["dataset_id"]
    run_root = config.RUNS_DIR / task_id
    run_root.mkdir(parents=True, exist_ok=True)

    # 每次训练独立子目录，避免 ultralytics 自动加版本号导致 best.pt 路径漂移
    run_dir = run_root / "run"
    if run_dir.exists():
        import shutil
        shutil.rmtree(run_dir, ignore_errors=True)

    weights = args.get("model", "yolo11n")
    if not str(weights).endswith(".pt"):
        weights = f"{weights}.pt"
    pretrained = args.get("pretrained", True)

    write_state(run_dir, status="starting", epoch=0, total_epochs=int(args.get("epochs", 50)), phase="准备中")
    db.update_task(task_id, status="running", started_at=db.now(), run_dir=str(run_dir))

    try:
        from ultralytics import YOLO
        import ultralytics
        ultralytics.settings["sync"] = False  # 关闭匿名统计上报

        # 基础权重优先级：仓库模型 / 本机权重(init_weights) > 内置模型名(需下载或已缓存)
        init_weights = (args.get("init_weights") or "").strip()
        if init_weights and Path(init_weights).exists():
            weights_path = str(Path(init_weights).resolve())
        elif Path(weights).exists():
            weights_path = str(Path(weights).resolve())
        else:
            local = config.MODELS_DIR / "pretrained" / weights
            if local.exists():
                weights_path = str(local)
            elif args.get("pretrained") is False:
                # 从头训练：用本地 yaml 配置
                weights_path = weights
            else:
                weights_path = inference_ensure(weights)
        print(f"[worker] 基础权重: {weights_path}"
              f"{'（来自模型仓库）' if init_weights else ''}", flush=True)

        _ensure_fonts(datasets.get_classes(ds_id))     # 中文字体预检，避免联网下载卡死

        # ---- 模型结构改进（视频方向二）：yaml 结构 + 预训练部分迁移 ----
        arch = [k for k in (args.get("arch") or []) if k]
        base_pt = weights_path                          # 已解析/缓存的基础 .pt
        train_pretrained: object = bool(pretrained)
        if arch:
            from app import model_zoo

            if model_zoo.supported(args.get("model", "")) and base_pt.endswith(".pt"):
                try:
                    nc = len(datasets.get_classes(ds_id)) or 1
                    weights_path = model_zoo.ensure_zoo_yaml(args["model"], arch, nc)
                    # 结构不同无法整包加载：yaml 建模 + pretrained 指向基础权重，
                    # ultralytics 按名称+形状迁移，新注意力层/P2 分支保持随机初始化
                    train_pretrained = base_pt if pretrained else False
                    print(f"[worker] 结构改进: {'+'.join(arch)} → {Path(weights_path).name}"
                          f"（迁移基础权重: {'是' if pretrained else '否，从头训练'}）", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[worker] 结构改进生成失败，回退原版结构: {e}", flush=True)
                    arch = []
            else:
                print("[worker] 所选基础模型不支持结构改进，忽略", flush=True)
                arch = []
        model = YOLO(weights_path)

        # ---- Loss 改进（视频方向三）：Wise-IoU / SIoU 替换 CIoU 回归项 ----
        iou_loss = (args.get("iou_loss") or "").strip().lower()
        if iou_loss:
            from app.losses import patch_bbox_iou

            if patch_bbox_iou(iou_loss):
                print(f"[worker] 回归损失已替换为 {iou_loss.upper()}", flush=True)

        # Windows 上 DataLoader 多进程代价高（每个 worker 重新 import torch）。
        # 小数据集用多进程反而更慢，这里按平台与数据量自动收敛。
        n_workers = int(args.get("workers", 0))
        stats = datasets.scan_stats(ds_id)
        if _IS_WIN and (stats["n_images"] < 200 or n_workers > 4):
            n_workers = min(n_workers, 2)
        if stats["n_images"] < 50:
            n_workers = 0

        train_kwargs = {
            "data": str(datasets.yaml_path(ds_id)),
            "epochs": int(args.get("epochs", 50)),
            "imgsz": int(args.get("imgsz", 640)),
            "batch": int(args.get("batch", 8)),
            "device": args.get("device", "cpu"),
            "workers": n_workers,
            "project": str(run_root),
            "name": "run",
            "exist_ok": True,
            "pretrained": train_pretrained,
            "verbose": True,
            "plots": True,
            "val": True,
        }
        if n_workers != int(args.get("workers", 0)):
            print(f"[worker] DataLoader workers 自动调整为 {n_workers}"
                  f"（原 {args.get('workers')}，图片 {stats['n_images']} 张）", flush=True)
        # 可选超参
        for k in ("lr0", "lrf", "mosaic", "mixup", "copy_paste", "hsv_h", "hsv_s", "hsv_v",
                  "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr",
                  "erasing", "patience", "seed", "close_mosaic", "dropout", "weight_decay",
                  "multi_scale", "warmup_epochs"):
            if args.get(k) not in (None, ""):
                train_kwargs[k] = float(args[k]) if k not in ("patience", "seed", "close_mosaic") else int(args[k])
        if args.get("cos_lr"):
            train_kwargs["cos_lr"] = True
        if args.get("optimizer"):
            train_kwargs["optimizer"] = args["optimizer"]
        # resume_from：以 last.pt 为初始权重继续训练（新 run 目录，产物独立）

        total = train_kwargs["epochs"]
        base_epoch = 0

        # ---- 回调：进度与逐轮指标 ----
        def on_epoch_end(trainer):
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            try:
                m = {k: float(v) for k, v in dict(trainer.metrics or {}).items()
                     if isinstance(v, (int, float)) or (hasattr(v, "item"))}
            except Exception:
                m = {}
            rec = {"epoch": epoch + base_epoch, "total": total + base_epoch, "metrics": m}
            try:
                # ultralytics 8.4: tloss 为滑动平均 loss dict，label_loss_items 加 train/ 前缀
                tl = getattr(trainer, "tloss", None)
                if tl is not None and hasattr(trainer, "label_loss_items"):
                    rec["loss"] = {k: float(v) for k, v in trainer.label_loss_items(tl, prefix="train").items()}
            except Exception:
                pass
            append_metric(run_dir, rec)
            write_state(run_dir, status="training", epoch=epoch + base_epoch,
                        total_epochs=total + base_epoch, phase="训练中")

        def on_pretrain_routine_start(trainer):
            write_state(run_dir, status="scanning", epoch=base_epoch,
                        total_epochs=total + base_epoch, phase="扫描数据集/缓存标签")

        def on_val_end(validator):
            write_state(run_dir, status="validating", phase="验证中")

        def on_train_end(trainer):
            write_state(run_dir, status="finished_train", phase="训练结束，保存权重")

        model.add_callback("on_pretrain_routine_start", on_pretrain_routine_start)
        model.add_callback("on_train_epoch_end", on_epoch_end)
        model.add_callback("on_val_end", on_val_end)
        model.add_callback("on_train_end", on_train_end)

        print(f"[worker] 开始训练 task={task_id} model={weights} epochs={total} "
              f"imgsz={train_kwargs['imgsz']} batch={train_kwargs['batch']} device={train_kwargs['device']}",
              flush=True)
        results = model.train(**train_kwargs)

        # ---- 定位 best.pt ----
        best = run_dir / "weights" / "best.pt"
        last = run_dir / "weights" / "last.pt"
        if not best.exists() and last.exists():
            best = last
        if not best.exists():
            for cand in run_root.rglob("best.pt"):
                best = cand
                break

        metrics: dict = {}
        try:
            box = results.box if hasattr(results, "box") else None
            if box is not None:
                metrics = {
                    "mAP50": float(box.map50),
                    "mAP50-95": float(box.map),
                    "precision": float(box.mp),
                    "recall": float(box.mr),
                }
        except Exception:
            pass
        # 兜底：取最后一轮 jsonl 指标
        if not metrics and _metrics_path(run_dir).exists():
            lines = _metrics_path(run_dir).read_text(encoding="utf-8").strip().splitlines()
            if lines:
                metrics = json.loads(lines[-1]).get("metrics", {})

        classes = datasets.get_classes(ds_id)
        task_kind = datasets.get_task(ds_id)
        if best.exists():
            db.add_model(
                name=f"{task['name']}",
                path=str(best), task_id=task_id, dataset_id=ds_id,
                base_model=task["model"], classes=classes, metrics=metrics,
                size=best.stat().st_size, source="trained", task=task_kind,
            )
            db.update_task(task_id, status="completed", finished_at=db.now(),
                           best_model=str(best), metrics=metrics, progress=1.0)
            write_state(run_dir, status="completed", phase="完成", metrics=metrics, best=str(best))
            print(f"[worker] 训练完成 best={best}", flush=True)
            return 0
        raise RuntimeError("训练结束但未找到 best.pt 权重文件")

    except KeyboardInterrupt:
        db.update_task(task_id, status="stopped", finished_at=db.now())
        write_state(run_dir, status="stopped", phase="已手动停止")
        print("[worker] 收到中断，已停止", flush=True)
        return 130
    except Exception as e:  # noqa: BLE001
        err = traceback.format_exc()
        db.update_task(task_id, status="failed", finished_at=db.now(), error=str(e)[:2000])
        write_state(run_dir, status="failed", phase="失败", error=str(e)[:2000])
        print(f"[worker] 训练失败: {e}\n{err}", flush=True)
        return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m app.train_worker <task_id>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
