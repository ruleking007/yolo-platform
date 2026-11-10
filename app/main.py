"""YOLO 训练平台 —— FastAPI 服务。

启动：  python -m app.server   （或 run.bat）
默认：  http://127.0.0.1:8750
"""
from __future__ import annotations

import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import capture, config, datasets, db, filedialog_native, inference, trainer
from . import images as images_mod

app = FastAPI(title="YOLO 训练平台", version="1.0")


# ================================================================ 系统

@app.get("/api/system")
def api_system():
    import psutil
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    try:
        import torch
        torch_ok = True
        cuda = torch.cuda.is_available()
        torch_ver = torch.__version__
    except Exception:
        torch_ok, cuda, torch_ver = False, False, "-"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": {"percent": cpu, "count": psutil.cpu_count(logical=True)},
        "memory": {"percent": mem.percent, "total_gb": round(mem.total / 2**30, 1),
                   "used_gb": round(mem.used / 2**30, 1)},
        "torch": {"installed": torch_ok, "version": torch_ver, "cuda": cuda,
                  "device": "cuda" if cuda else "cpu"},
        "datasets": len(list(config.DATASETS_DIR.iterdir())),
        "tasks": len(db.list_tasks()),
        "models": len(db.list_models()),
    }


@app.get("/api/models/choices")
def api_model_choices():
    return {"models": config.MODEL_CHOICES, "exports": config.EXPORT_FORMATS,
            "pretrained": inference.pretrained_list()}


@app.post("/api/models/pretrained")
def api_download_pretrained(payload: dict = Body(...)):
    mid = payload.get("model", "yolo11n")
    try:
        path = inference.ensure_pretrained(mid)
        return {"ok": True, "path": path}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"下载失败: {e}")


# ================================================================ 数据集

@app.get("/api/datasets")
def api_list_datasets():
    return datasets.list_datasets()


@app.post("/api/datasets")
def api_create_dataset(payload: dict = Body(...)):
    name = (payload.get("name") or "").strip()
    classes = [c.strip() for c in payload.get("classes", []) if str(c).strip()]
    if not name or not classes:
        raise HTTPException(400, "数据集名称与类别列表不能为空")
    ds_id = datasets.create_empty(name, classes, payload.get("task", "detect"))
    return {"id": ds_id}


@app.post("/api/datasets/upload")
async def api_upload_dataset(file: UploadFile = File(...),
                             name: str = Form(""),
                             task: str = Form("detect"),
                             split_ratio: float = Form(0.8)):
    # 流式落盘，避免大 zip 占内存
    suffix = Path(file.filename or "upload").suffix.lower() or ".bin"
    fd, tmp_name = tempfile.mkstemp(suffix=suffix, dir=str(config.TMP_DIR))
    try:
        with open(fd, "wb") as fh:
            while chunk := await file.read(1 << 20):
                fh.write(chunk)
        try:
            with open(tmp_name, "rb") as fh:
                ds_id = datasets.import_upload(file.filename or "upload", fh,
                                               name=name or None, task=task,
                                               split_ratio=split_ratio)
            return {"id": ds_id, **datasets.scan_stats(ds_id)}
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"导入失败: {e}")
    finally:
        Path(tmp_name).unlink(missing_ok=True)


# ---------- 纯图片流程：客户只给图片 ----------

@app.post("/api/datasets/from-images")
def api_create_from_images(payload: dict = Body(...)):
    """第一步：建数据集骨架（只需名称；类别可留空，标注时再补）。"""
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "请填写数据集名称")
    ds_id = datasets.create_from_images(
        name,
        classes=payload.get("classes") or None,
        task=payload.get("task", "detect"),
    )
    return {"id": ds_id, "name": name, **datasets.scan_stats(ds_id)}


@app.post("/api/datasets/{ds_id}/ingest")
async def api_ingest_images(ds_id: str,
                            files: list[UploadFile] = File(...),
                            split: str = Form("train"),
                            max_side: int = Form(1920),
                            dedupe: bool = Form(True)):
    """第二步：分批上传图片（前端按批调用，边传边入库）。"""
    _require_ds(ds_id)
    if split not in ("train", "val"):
        raise HTTPException(400, "split 非法")
    if len(files) > 300:
        raise HTTPException(400, "单批最多 300 张，请分批上传")

    staged: list[tuple[str, Path]] = []
    tmp_root = Path(tempfile.mkdtemp(dir=str(config.TMP_DIR)))
    try:
        for f in files:
            suffix = Path(f.filename or "x").suffix.lower() or ".bin"
            fd, name = tempfile.mkstemp(suffix=suffix, dir=str(tmp_root))
            with open(fd, "wb") as fh:
                while chunk := await f.read(1 << 18):
                    fh.write(chunk)
            staged.append((f.filename or "image", Path(name)))
        res = datasets.ingest_images(ds_id, staged, split=split,
                                     max_side=max_side, dedupe=dedupe)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return {**res, **datasets.scan_stats(ds_id)}


@app.post("/api/datasets/{ds_id}/finalize")
def api_finalize_dataset(ds_id: str, payload: dict = Body(default={})):
    """第三步：定类别、自动划分 train/val、写 data.yaml。"""
    _require_ds(ds_id)
    classes = payload.get("classes") or None
    stats = datasets.finalize_dataset(
        ds_id, classes=classes,
        split_ratio=float(payload.get("split_ratio", 0.8)),
        task=payload.get("task"), name=payload.get("name"),
    )
    return {"ok": True, **stats}


@app.post("/api/datasets/{ds_id}/ingest-paths")
def api_ingest_paths(ds_id: str, payload: dict = Body(...)):
    """按本机绝对路径导入图片（配合系统文件对话框），走与网页上传相同的规范化流程。"""
    _require_ds(ds_id)
    paths = payload.get("paths") or []
    if not isinstance(paths, list) or not paths:
        raise HTTPException(400, "请提供图片路径列表")
    if len(paths) > 300:
        raise HTTPException(400, "单次最多 300 张")
    split = payload.get("split", "train")
    if split not in ("train", "val"):
        raise HTTPException(400, "split 非法")

    staged: list[tuple[str, Path]] = []
    skipped: list[str] = []
    for raw in paths:
        p = Path(str(raw)).expanduser()
        if not p.is_absolute():
            p = (config.BASE_DIR / str(raw)).resolve()
        if not p.exists() or not p.is_file():
            skipped.append(f"{p}（不存在）")
            continue
        if p.suffix.lower() not in config.IMAGE_EXTS:
            skipped.append(f"{p.name}（不是图片）")
            continue
        if p.stat().st_size > images_mod.MAX_FILE_BYTES:
            skipped.append(f"{p.name}（过大）")
            continue
        staged.append((p.name, p))

    if not staged:
        raise HTTPException(400, "没有可导入的图片：" + "；".join(skipped[:5]))

    res = datasets.ingest_images(ds_id, staged, split=split,
                                 max_side=int(payload.get("max_side", 1920)),
                                 dedupe=bool(payload.get("dedupe", True)))
    return {**res, "skipped_paths": skipped[:50], **datasets.scan_stats(ds_id)}


@app.post("/api/datasets/{ds_id}/detect-classes")
def api_detect_classes(ds_id: str):
    """扫描现有标注，返回实际使用到的类别编号，提示数据集配置是否有缺口。"""
    _require_ds(ds_id)
    used: set[int] = set()
    for split in ("train", "val"):
        d = datasets.labels_dir(ds_id) / split
        if not d.exists():
            continue
        for lp in d.glob("*.txt"):
            for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        used.add(int(float(parts[0])))
                    except ValueError:
                        pass
    configured = datasets.get_classes(ds_id)
    return {"used_classes": sorted(used), "configured": configured,
            "nc": len(configured), "max_used": max(used) if used else -1,
            "ok": (max(used) if used else -1) < len(configured)}


@app.get("/api/datasets/{ds_id}")
def api_dataset_detail(ds_id: str):
    _require_ds(ds_id)
    return {
        "id": ds_id,
        **datasets.read_meta(ds_id),
        "classes": datasets.get_classes(ds_id),
        "task": datasets.get_task(ds_id),
        **datasets.scan_stats(ds_id),
    }


@app.get("/api/datasets/{ds_id}/images")
def api_dataset_images(ds_id: str, split: str = Query(None), q: str = Query(""),
                       labeled: str = Query(None), offset: int = 0, limit: int = 60):
    _require_ds(ds_id)
    return datasets.list_images(ds_id, split, q, labeled, offset, limit)


@app.get("/api/datasets/{ds_id}/thumb/{split}/{name}")
def api_thumb(ds_id: str, split: str, name: str):
    _require_ds(ds_id)
    try:
        data = datasets.thumbnail(ds_id, split, name)
    except Exception:
        raise HTTPException(404, "图片不存在")
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=300"})


@app.get("/api/datasets/{ds_id}/image/{split}/{name}")
def api_image(ds_id: str, split: str, name: str, full: int = 0):
    _require_ds(ds_id)
    try:
        data = datasets.full_sized(ds_id, split, name) if not full else \
            datasets.image_path(ds_id, split, name).read_bytes()
    except Exception:
        raise HTTPException(404, "图片不存在")
    return Response(content=data, media_type="image/jpeg")


@app.get("/api/datasets/{ds_id}/label/{split}/{stem}")
def api_get_label(ds_id: str, split: str, stem: str):
    _require_ds(ds_id)
    return {"boxes": datasets.read_shapes(ds_id, split, stem),
            "task": datasets.get_task(ds_id)}


@app.post("/api/datasets/{ds_id}/label/{split}/{stem}")
async def api_put_label(ds_id: str, split: str, stem: str, payload: dict = Body(...)):
    _require_ds(ds_id)
    if split not in ("train", "val"):
        raise HTTPException(400, "split 非法")
    boxes = payload.get("boxes", [])
    datasets.write_label(ds_id, split, stem, boxes, datasets.get_task(ds_id))
    return {"ok": True, "n": len(boxes)}


@app.post("/api/datasets/{ds_id}/images/add")
async def api_add_images(ds_id: str, files: list[UploadFile] = File(...),
                         split: str = Form("train")):
    _require_ds(ds_id)
    batch: list[tuple[str, bytes]] = []
    for f in files[:500]:
        data = await f.read()
        if len(data) < 10_000_000:
            batch.append((f.filename or "img.jpg", data))
    n = datasets.add_images(ds_id, batch, split)
    return {"added": n}


@app.post("/api/datasets/{ds_id}/split")
def api_resplit(ds_id: str, payload: dict = Body(...)):
    _require_ds(ds_id)
    stats = datasets.split_existing(ds_id, float(payload.get("ratio", 0.8)))
    return {"ok": True, **stats}


@app.put("/api/datasets/{ds_id}/classes")
def api_set_classes(ds_id: str, payload: dict = Body(...)):
    _require_ds(ds_id)
    classes = [str(c).strip() for c in payload.get("classes", []) if str(c).strip()]
    if not classes:
        raise HTTPException(400, "类别不能为空")
    datasets.set_classes(ds_id, classes)
    return {"ok": True, "classes": classes}


@app.put("/api/datasets/{ds_id}/rename")
def api_rename_ds(ds_id: str, payload: dict = Body(...)):
    _require_ds(ds_id)
    datasets.rename_dataset(ds_id, payload.get("name", ""))
    return {"ok": True}


@app.delete("/api/datasets/{ds_id}/image/{split}/{name}")
def api_delete_image(ds_id: str, split: str, name: str):
    _require_ds(ds_id)
    datasets.delete_image(ds_id, split, name)
    return {"ok": True}


@app.delete("/api/datasets/{ds_id}")
def api_delete_dataset(ds_id: str):
    _require_ds(ds_id)
    datasets.delete_dataset(ds_id)
    return {"ok": True}


@app.get("/api/datasets/{ds_id}/export")
def api_export_dataset(ds_id: str):
    _require_ds(ds_id)
    try:
        z = datasets.export_zip(ds_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))
    meta = datasets.read_meta(ds_id)
    fname = f"{meta.get('name', ds_id)}.zip"
    return FileResponse(z, filename=fname, media_type="application/zip")


def _require_ds(ds_id: str) -> None:
    if not datasets.ds_dir(ds_id).exists() or ".." in ds_id or "/" in ds_id or "\\" in ds_id:
        raise HTTPException(404, "数据集不存在")


# ================================================================ 演示数据

@app.post("/api/demo")
def api_demo(payload: dict = Body(default={})):
    ds_id = inference.make_demo_dataset(
        name=payload.get("name", "demo-shapes"),
        n=int(payload.get("count", 120)))
    return {"id": ds_id, **datasets.scan_stats(ds_id)}


# ================================================================ 训练任务

@app.get("/api/tasks")
def api_tasks():
    out = []
    for t in db.list_tasks():
        t["progress"] = trainer.progress_of(t)
        t.pop("log_path", None)
        out.append(t)
    return out


@app.post("/api/tasks")
def api_start_task(payload: dict = Body(...)):
    ds_id = payload.get("dataset_id")
    if not ds_id or not datasets.ds_dir(ds_id).exists():
        raise HTTPException(400, "请先选择有效的数据集")
    stats = datasets.scan_stats(ds_id)
    if stats["n_images"] == 0:
        raise HTTPException(400, "数据集里没有图片")
    if stats["n_labeled"] == 0:
        raise HTTPException(400, "数据集里没有已标注的图片，请先标注")
    name = (payload.get("name") or f"train-{ds_id[:8]}").strip()

    # 基础权重来源：仓库模型(init_model_id) > 本机权重(init_weights) > 内置模型(model)
    init_mid = payload.get("init_model_id")
    init_weights = (payload.get("init_weights") or "").strip()
    model = payload.get("model", "yolo11n")
    resolved = None
    warn = None

    if init_mid:
        m = db.get_model(init_mid)
        if not m or not Path(m["path"]).exists():
            raise HTTPException(400, "所选仓库模型不存在或权重文件丢失")
        resolved = m["path"]
        model = m["name"]
        # 类别数不一致时仍可训练（YOLO 会重建分类头），但要让用户知情
        ds_classes = datasets.get_classes(ds_id)
        if m.get("classes") and len(m["classes"]) != len(ds_classes):
            warn = (f"所选模型有 {len(m['classes'])} 个类别，当前数据集有 {len(ds_classes)} 个，"
                    f"训练时会重建分类头（相当于部分迁移学习）")
        else:
            warn = None
    elif init_weights:
        p = Path(init_weights).expanduser()
        if not p.is_absolute():
            p = (config.BASE_DIR / init_weights).resolve()
        if not p.exists() or not p.is_file():
            raise HTTPException(400, f"初始权重文件不存在：{p}")
        resolved = str(p)
        model = p.stem

    args = {k: v for k, v in payload.items()
            if k in ("epochs", "imgsz", "batch", "device", "workers", "pretrained", "optimizer",
                     "lr0", "lrf", "patience", "seed", "mosaic", "mixup", "copy_paste",
                     "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr",
                     "erasing", "hsv_h", "hsv_s", "hsv_v", "close_mosaic",
                     "weight_decay", "dropout", "resume_from",
                     "arch", "iou_loss", "multi_scale", "cos_lr", "warmup_epochs")}
    # 结构改进 / loss 改进校验
    from app import model_zoo

    arch = payload.get("arch") or []
    if isinstance(arch, str):
        arch = [arch]
    arch = [k for k in arch if k in model_zoo.ARCH_KEYS]
    if arch and not model_zoo.supported(model):
        raise HTTPException(400, f"结构改进仅支持内置 yolo11/yolov8（n/s/m），当前基础模型：{model}")
    args["arch"] = arch
    iou_loss = (payload.get("iou_loss") or "").strip().lower()
    if iou_loss and iou_loss not in model_zoo.IOU_LOSS_KEYS:
        raise HTTPException(400, f"不支持的损失函数：{iou_loss}")
    args["iou_loss"] = iou_loss
    args.setdefault("epochs", 30)
    args.setdefault("imgsz", 480)
    args.setdefault("batch", 8)
    args.setdefault("device", "cpu")
    args.setdefault("workers", 0)
    args["model"] = model
    if resolved:
        args["init_weights"] = resolved          # worker 优先用这个绝对路径
        args["pretrained"] = True
    if init_mid:
        args["init_model_id"] = init_mid

    task_id = db.create_task(name, ds_id, model, args, trainer._log_path("placeholder"))
    # 修正日志路径到任务目录
    db.update_task(task_id, log_path=trainer._log_path(task_id))
    trainer.start(task_id)
    return {"id": task_id, "init_weights": resolved, "warning": warn}


@app.get("/api/tasks/{task_id}")
def api_task_detail(task_id: str):
    t = db.get_task(task_id)
    if not t:
        raise HTTPException(404, "任务不存在")
    t["progress"] = trainer.progress_of(t)
    t["metrics"] = json.loads(t.get("metrics") or "{}")
    t["args"] = json.loads(t.get("args") or "{}")
    t["state"] = trainer.read_state(task_id)
    return t


@app.get("/api/tasks/{task_id}/metrics")
def api_task_metrics(task_id: str):
    if not db.get_task(task_id):
        raise HTTPException(404, "任务不存在")
    return trainer.read_metrics(task_id)


@app.get("/api/train/improvements")
def api_train_improvements():
    """结构改进 / 损失函数选项（视频三大方向之「模型改进」「loss 优化」落地清单）。"""
    from app import model_zoo

    return {"arch": model_zoo.ARCH_OPTIONS, "iou_loss": model_zoo.IOU_LOSS_OPTIONS}


@app.get("/api/tasks/{task_id}/log")
def api_task_log(task_id: str, offset: int = 0):
    if not db.get_task(task_id):
        raise HTTPException(404, "任务不存在")
    return trainer.tail_log(task_id, offset)


@app.post("/api/tasks/{task_id}/stop")
def api_stop_task(task_id: str):
    if not db.get_task(task_id):
        raise HTTPException(404, "任务不存在")
    return {"ok": trainer.stop(task_id)}


@app.delete("/api/tasks/{task_id}")
def api_delete_task(task_id: str):
    trainer.delete(task_id)
    return {"ok": True}


@app.post("/api/tasks/{task_id}/resume")
def api_resume_task(task_id: str):
    """基于原任务的 last.pt 续训（新任务）。"""
    old = db.get_task(task_id)
    if not old:
        raise HTTPException(404, "任务不存在")
    run_dir = old.get("run_dir")
    last = Path(run_dir) / "weights" / "last.pt" if run_dir else None
    if not last or not last.exists():
        raise HTTPException(400, "没有找到 last.pt，无法续训")
    args = json.loads(old.get("args") or "{}")
    args["resume_from"] = str(last)
    args["model"] = str(last)
    name = old["name"] + "-resume"
    new_id = db.create_task(name, old["dataset_id"], old["model"], args, "")
    db.update_task(new_id, log_path=trainer._log_path(new_id))
    trainer.start(new_id)
    return {"id": new_id}


# ================================================================ 任务产物（曲线图/混淆矩阵等）

@app.get("/api/tasks/{task_id}/artifacts")
def api_artifacts(task_id: str):
    t = db.get_task(task_id)
    if not t or not t.get("run_dir"):
        return []
    run = Path(t["run_dir"])
    out = []
    for p in sorted(run.glob("*.png")):
        out.append({"name": p.name, "url": f"/api/tasks/{task_id}/artifact/{quote(p.name)}"})
    return out


@app.get("/api/tasks/{task_id}/artifact/{name}")
def api_artifact(task_id: str, name: str):
    t = db.get_task(task_id)
    if not t or not t.get("run_dir"):
        raise HTTPException(404)
    p = (Path(t["run_dir"]) / name).resolve()
    if Path(t["run_dir"]).resolve() not in p.parents or p.suffix != ".png":
        raise HTTPException(400, "非法路径")
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


# ================================================================ 模型仓库

@app.get("/api/models")
def api_models():
    out = []
    for m in db.list_models():
        p = Path(m["path"])
        m["exists"] = p.exists()
        m["size"] = p.stat().st_size if p.exists() else 0
        m["in_repo"] = _is_in_repo(p)
        out.append(m)
    return out


def _is_in_repo(p: Path) -> bool:
    roots = [config.MODELS_DIR.resolve(), config.RUNS_DIR.resolve()]
    try:
        rp = p.resolve()
    except Exception:
        return False
    return any(r == rp or r in rp.parents for r in roots)


# ---------- 从本机添加模型 ----------

@app.post("/api/models/import-local")
def api_import_local(payload: dict = Body(...)):
    """把本机磁盘上的权重文件登记进模型仓库。

    两种模式：
      - copy: 复制到 data/models/imported/（推荐，避免原文件被移动/删除后失效）
      - link: 仅登记原路径（大文件或想保持单一副本时用）
    """
    raw = (payload.get("path") or "").strip().strip('"')
    if not raw:
        raise HTTPException(400, "请提供权重文件路径")
    src = Path(raw).expanduser()
    if not src.is_absolute():
        # 相对路径按平台根目录解析
        src = (config.BASE_DIR / raw).resolve()
    if not src.exists() or not src.is_file():
        raise HTTPException(400, f"文件不存在：{src}")
    if src.suffix.lower() not in inference.WEIGHT_EXTS:
        raise HTTPException(400, f"不支持的权重格式 {src.suffix}（支持 {', '.join(sorted(inference.WEIGHT_EXTS))}）")
    if src.stat().st_size < 20_000:
        raise HTTPException(400, f"文件过小（{src.stat().st_size} 字节），可能不是有效权重")

    mode = payload.get("mode", "copy")
    if mode not in ("copy", "link"):
        raise HTTPException(400, "mode 只能是 copy 或 link")

    name = (payload.get("name") or "").strip() or src.stem
    final = src
    if mode == "copy":
        dest_dir = config.MODELS_DIR / "imported"
        dest_dir.mkdir(parents=True, exist_ok=True)
        final = _unique_file(dest_dir, src.name)
        shutil.copy2(src, final)

    # 自动读元信息（.pt 能拿到类别/任务；其它格式留空由用户填）
    info = inference.inspect_weights(final)
    classes = payload.get("classes") or info.get("classes") or []
    task = payload.get("task") or info.get("task") or "detect"

    mid = db.add_model(
        name=name, path=str(final), classes=[str(c) for c in classes],
        base_model=payload.get("base_model") or None,
        source="uploaded", size=final.stat().st_size,
        task=task, note=payload.get("note") or f"来自本机：{src}",
    )
    return {"id": mid, "path": str(final), "classes": classes, "task": task,
            "size": final.stat().st_size, "mode": mode,
            "info": {k: v for k, v in info.items() if k != "classes"}}


@app.post("/api/models/inspect")
def api_inspect_weights(payload: dict = Body(...)):
    """只探测本机权重文件的元信息，不入库（供前端预览后再确认）。"""
    raw = (payload.get("path") or "").strip().strip('"')
    if not raw:
        raise HTTPException(400, "请提供文件路径")
    src = Path(raw).expanduser()
    if not src.is_absolute():
        src = (config.BASE_DIR / raw).resolve()
    if not src.exists() or not src.is_file():
        raise HTTPException(404, f"文件不存在：{src}")
    info = inference.inspect_weights(src)
    return {"path": str(src), "name": src.stem, "size": src.stat().st_size,
            "ext": src.suffix.lower(), **info}


@app.get("/api/models/browse")
def api_browse_dir(path: str = Query("")):
    """浏览本机目录，供「从本机添加」选择文件（只列出目录与权重文件）。"""
    target = Path(path).expanduser() if path else Path.home()
    if not target.exists() or not target.is_dir():
        raise HTTPException(404, f"目录不存在：{target}")
    items = []
    try:
        for p in sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if p.name.startswith(".") or p.name.startswith("__"):
                continue
            try:
                if p.is_dir():
                    items.append({"name": p.name, "path": str(p), "type": "dir"})
                elif p.suffix.lower() in inference.WEIGHT_EXTS and p.stat().st_size > 20_000:
                    items.append({"name": p.name, "path": str(p), "type": "file",
                                  "size": p.stat().st_size})
            except (PermissionError, OSError):
                continue
    except PermissionError:
        raise HTTPException(403, f"没有权限访问：{target}")
    return {"cwd": str(target), "parent": str(target.parent) if target.parent != target else None,
            "home": str(Path.home()), "items": items[:300]}


def _unique_file(dst_dir: Path, filename: str) -> Path:
    p = dst_dir / filename
    if not p.exists():
        return p
    stem, suf = p.stem, p.suffix
    i = 1
    while (dst_dir / f"{stem}_{i}{suf}").exists():
        i += 1
    return dst_dir / f"{stem}_{i}{suf}"


# ---------- 系统原生「打开文件」对话框 ----------

@app.get("/api/native-dialog/available")
def api_native_dialog_available():
    """前端据此决定是否显示「从资源管理器选择」按钮。"""
    ok, reason = filedialog_native.available()
    return {"available": ok, "reason": reason,
            "platform": sys.platform,
            "home": str(Path.home())}


@app.post("/api/native-dialog/pick")
def api_native_dialog_pick(payload: dict = Body(default={})):
    """弹出操作系统原生文件选择框，返回真实绝对路径。

    注意：浏览器沙箱拿不到真实路径，因此必须由服务端调起系统对话框。
    前端应在点击后立刻显示「请在弹出的系统窗口中选择…」提示。
    """
    kind = payload.get("kind", "file")          # file|dir
    purpose = payload.get("purpose", "weights")  # weights|images|any
    multiple = bool(payload.get("multiple", False))
    initialdir = payload.get("initialdir") or None
    timeout = float(payload.get("timeout", 180))

    filters = {
        "weights": filedialog_native.WEIGHT_FILTERS,
        "images": filedialog_native.IMAGE_FILTERS,
        "any": filedialog_native.ANY_FILTERS,
    }.get(purpose, filedialog_native.WEIGHT_FILTERS)

    titles = {"weights": "选择模型权重文件", "images": "选择图片文件", "any": "选择文件"}
    res = filedialog_native.pick_files(
        filters=filters,
        title=titles.get(purpose, "选择文件"),
        initialdir=initialdir,
        multiple=multiple,
        timeout=timeout,
        kind=kind,
    )
    if res.get("unsupported"):
        return {"ok": False, "paths": [], "unsupported": True,
                "error": f"当前运行环境无法弹出系统对话框（{res.get('error')}）。"
                         f"请改用「手填路径」或网页上传方式。"}
    if res.get("error") and not res.get("cancelled"):
        return {"ok": False, "paths": [], "error": res["error"]}
    return {"ok": res.get("ok", False), "paths": res.get("paths", []),
            "cancelled": res.get("cancelled", False)}


@app.get("/api/models/{model_id}")
def api_model_detail(model_id: str):
    m = db.get_model(model_id)
    if not m:
        raise HTTPException(404, "模型不存在")
    return m


@app.delete("/api/models/{model_id}")
def api_delete_model(model_id: str):
    m = db.get_model(model_id)
    if not m:
        raise HTTPException(404, "模型不存在")
    p = Path(m["path"])
    # 只删除位于平台目录内的权重文件
    roots = [config.MODELS_DIR.resolve(), config.RUNS_DIR.resolve()]
    rp = p.resolve()
    if any(r in rp.parents for r in roots):
        p.unlink(missing_ok=True)
    db.delete_model(model_id)
    return {"ok": True}


@app.get("/api/models/{model_id}/download")
def api_download_model(model_id: str):
    m = db.get_model(model_id)
    if not m or not Path(m["path"]).exists():
        raise HTTPException(404, "权重文件不存在")
    return FileResponse(m["path"], filename=f"{m['name']}.pt", media_type="application/octet-stream")


@app.post("/api/models/{model_id}/export")
def api_export(model_id: str, payload: dict = Body(...)):
    m = db.get_model(model_id)
    if not m or not Path(m["path"]).exists():
        raise HTTPException(404, "权重文件不存在")
    fmt = payload.get("format", "onnx")
    if fmt not in config.EXPORT_FORMATS:
        raise HTTPException(400, f"不支持的导出格式: {fmt}")
    try:
        res = inference.export_model(m["path"], fmt,
                                     imgsz=int(payload.get("imgsz", 640)),
                                     simplify=bool(payload.get("simplify", True)))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"导出失败: {e}")
    out_path = Path(res["path"])
    if out_path.exists() and config.MODELS_DIR.resolve() not in out_path.resolve().parents:
        # 拷贝进模型仓库方便下载
        dest = config.MODELS_DIR / out_path.name
        shutil.copy2(out_path, dest)
        res["path"] = str(dest)
        res["size"] = dest.stat().st_size
    return res


@app.get("/api/models/{model_id}/export-file")
def api_export_file(model_id: str):
    m = db.get_model(model_id)
    if not m:
        raise HTTPException(404)
    stem = Path(m["path"]).with_suffix("")
    for suf in (".onnx", ".torchscript", "_saved_model", ".engine"):
        cand = Path(str(stem) + suf)
        if cand.exists():
            if cand.is_dir():
                z = config.TMP_DIR / f"{model_id}{suf}.zip"
                shutil.make_archive(str(z).removesuffix(".zip"), "zip", cand)
                return FileResponse(z, filename=z.name)
            return FileResponse(cand, filename=cand.name)
    raise HTTPException(404, "尚未导出该格式")


# ================================================================ 推理测试

@app.post("/api/predict")
async def api_predict(model_id: str = Form(...), file: UploadFile = File(...),
                      conf: float = Form(0.25), imgsz: int = Form(640)):
    m = db.get_model(model_id)
    if not m or not Path(m["path"]).exists():
        raise HTTPException(404, "模型不存在")
    data = await file.read()
    if len(data) > 20_000_000:
        raise HTTPException(400, "图片过大（>20MB）")
    try:
        res = inference.predict(m["path"], data, conf=conf, imgsz=imgsz)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"推理失败: {e}")
    annotated = res.pop("annotated", None)
    token = db.new_id("img_")
    if annotated:
        (config.TMP_DIR / f"{token}.jpg").write_bytes(annotated)
    return {"result": res, "image_token": token if annotated else None}


@app.post("/api/predict/by-path")
def api_predict_by_path(payload: dict = Body(...)):
    """用本机绝对路径的图片直接推理（配合系统文件对话框使用）。"""
    model_id = payload.get("model_id")
    m = db.get_model(model_id) if model_id else None
    if not m or not Path(m["path"]).exists():
        raise HTTPException(404, "模型不存在")
    raw = (payload.get("path") or "").strip().strip('"')
    if not raw:
        raise HTTPException(400, "请提供图片路径")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (config.BASE_DIR / raw).resolve()
    if not p.exists() or not p.is_file():
        raise HTTPException(404, f"图片不存在：{p}")
    if p.stat().st_size > 30_000_000:
        raise HTTPException(400, "图片过大（>30MB）")
    try:
        res = inference.predict(m["path"], p.read_bytes(),
                                conf=float(payload.get("conf", 0.25)),
                                imgsz=int(payload.get("imgsz", 640)))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"推理失败: {e}")
    annotated = res.pop("annotated", None)
    token = db.new_id("img_")
    if annotated:
        (config.TMP_DIR / f"{token}.jpg").write_bytes(annotated)
    res["source"] = str(p)
    return {"result": res, "image_token": token if annotated else None}


@app.get("/api/predict/image/{token}")
def api_predict_image(token: str):
    if not (token.startswith("img_") and token.replace("_", "").isalnum()):
        raise HTTPException(400, "非法 token")
    p = (config.TMP_DIR / f"{token}.jpg").resolve()
    if config.TMP_DIR.resolve() not in p.parents or not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg")


@app.post("/api/predict/dataset-image")
def api_predict_ds_image(payload: dict = Body(...)):
    """用模型测试数据集中的某张图（返回 YOLO 格式结果供标注页对照）。"""
    m = db.get_model(payload["model_id"]) if payload.get("model_id") else None
    if not m or not Path(m["path"]).exists():
        raise HTTPException(404, "模型不存在")
    ds_id, split, name = payload["dataset_id"], payload["split"], payload["name"]
    _require_ds(ds_id)
    try:
        res = inference.predict_on_dataset_image(m["path"], ds_id, split, name,
                                                 conf=float(payload.get("conf", 0.25)))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"推理失败: {e}")
    # 转成 YOLO 归一化框，便于前端直接显示
    w, h = res["width"], res["height"]
    boxes = []
    for d in res["detections"]:
        x1, y1, x2, y2 = d["bbox"]
        boxes.append({"cls": d["cls"], "conf": d["conf"],
                      "cx": (x1 + x2) / 2 / w, "cy": (y1 + y2) / 2 / h,
                      "w": (x2 - x1) / w, "h": (y2 - y1) / h})
    return {"boxes": boxes, "n": len(boxes), "speed_ms": res["speed"]}


# ================================================================ 屏幕采集插件

@app.get("/api/capture/capability")
def api_capture_capability():
    ok, reason = capture.capture_available()
    w, h = capture.screen_size()
    pos = capture.cursor_pos()
    return {"available": ok, "reason": reason, "platform": sys.platform,
            "screen": [w, h], "cursor": list(pos) if pos else None}


@app.post("/api/capture/test")
def api_capture_test():
    """立即截一张图入库（验证插件可用），返回识别到的信息。"""
    if not capture.capture_available()[0]:
        raise HTTPException(400, "当前环境无法截屏")
    ds_id = (db.get_setting("capture_test_ds") or "")
    if not ds_id or not datasets.ds_dir(ds_id).exists():
        ds_id = datasets.create_from_images("屏幕采集-测试", ["object"])
        db.set_setting("capture_test_ds", ds_id)
    tmpdir = config.TMP_DIR / "cap_test"
    fp = capture.grab_screen(tmpdir)
    if fp is None:
        raise HTTPException(500, "截屏失败")
    try:
        res = datasets.ingest_images(ds_id, [(fp.name, fp)], split="train", dedupe=True)
    finally:
        fp.unlink(missing_ok=True)
        shutil.rmtree(tmpdir, ignore_errors=True)
    token = db.new_id("img_")
    src = datasets.images_dir(ds_id) / "train" / f"{res['stems'][0]}.jpg" if res["stems"] else None
    if src and src.exists():
        (config.TMP_DIR / f"{token}.jpg").write_bytes(src.read_bytes())
    return {"dataset_id": ds_id, "added": res["added"],
            "duplicate": res["skipped_duplicate"], "image_token": token if res["added"] else None}


@app.get("/api/capture/jobs")
def api_capture_jobs():
    return [capture.job_status(j) for j in db.list_capture_jobs()]


@app.post("/api/capture/jobs")
def api_capture_create(payload: dict = Body(...)):
    ds_id = payload.get("dataset_id")
    if not ds_id or not datasets.ds_dir(ds_id).exists():
        raise HTTPException(400, "请选择有效的数据集（采集的图片会存入该数据集）")
    try:
        job = capture.create_job(
            dataset_id=ds_id,
            name=payload.get("name") or "屏幕采集",
            mode=payload.get("mode", "once"),
            at=payload.get("at", ""),
            duration_min=payload.get("duration_min", 10),
            interval_sec=payload.get("interval_sec", 1),
            active_window=payload.get("active_window", 5),
            move_threshold=payload.get("move_threshold", 3),
            max_images=payload.get("max_images", 600),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return capture.job_status(job)


@app.post("/api/capture/jobs/{job_id}/start")
def api_capture_start(job_id: str):
    """恢复排程（到点自动跑）。"""
    job = db.get_capture_job(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    capture.reschedule(job_id)
    return {"ok": True, "next_run": db.get_capture_job(job_id).get("next_run")}


@app.post("/api/capture/jobs/{job_id}/run-now")
def api_capture_run_now(job_id: str):
    job = db.get_capture_job(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    started = capture.start_job(job_id)
    if not started:
        raise HTTPException(400, "任务已在运行中")
    return {"ok": True}


@app.post("/api/capture/jobs/{job_id}/stop")
def api_capture_stop(job_id: str):
    if not db.get_capture_job(job_id):
        raise HTTPException(404, "任务不存在")
    return {"ok": capture.stop_job(job_id)}


@app.post("/api/capture/jobs/{job_id}/resume")
def api_capture_resume(job_id: str):
    job = db.get_capture_job(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    ok = capture.reschedule(job_id)
    return {"ok": ok, "next_run": db.get_capture_job(job_id).get("next_run")}


@app.delete("/api/capture/jobs/{job_id}")
def api_capture_delete(job_id: str):
    capture.stop_job(job_id)
    db.delete_capture_job(job_id)
    return {"ok": True}


@app.get("/api/capture/jobs/{job_id}/recent")
def api_capture_recent(job_id: str):
    job = db.get_capture_job(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return capture.recent_captures(job["dataset_id"])


# ================================================================ 静态页面

if config.WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(config.WEB_DIR), html=True), name="web")


@app.exception_handler(Exception)
async def unhandled_exc_handler(request, exc):  # noqa: ARG001
    return JSONResponse({"detail": f"内部错误: {exc}"}, status_code=500)
