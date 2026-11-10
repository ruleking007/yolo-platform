"""推理预览、模型导出与演示数据集生成。"""
from __future__ import annotations

import io
import random
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

from . import config, db, datasets


# ------------------------------------------------------------------ 推理

def predict(model_path: str, image_bytes: bytes, conf: float = 0.25,
            iou: float = 0.7, imgsz: int = 640, max_det: int = 300) -> dict:
    """对单张图片推理，返回标注结果 + 画框后的 JPEG。"""
    import tempfile
    from ultralytics import YOLO

    model = YOLO(model_path)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=str(config.TMP_DIR)) as tf:
        tf.write(image_bytes)
        tmp_in = Path(tf.name)
    try:
        res = model.predict(source=str(tmp_in), conf=conf, iou=iou, imgsz=imgsz,
                            max_det=max_det, verbose=False, device="cpu")
        r = res[0]
        dets = []
        names = r.names if isinstance(r.names, dict) else {i: n for i, n in enumerate(r.names)}
        boxes = getattr(r, "boxes", None)
        if boxes is not None:
            for b in boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                dets.append({
                    "cls": int(b.cls[0]),
                    "cls_name": str(names.get(int(b.cls[0]), int(b.cls[0]))),
                    "conf": round(float(b.conf[0]), 4),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                })
        annotated = None
        if hasattr(r, "plot"):
            im = r.plot(line_width=2, font_size=12)
            img = Image.fromarray(im[:, :, ::-1])  # BGR->RGB
            if max(img.size) > 1400:
                img.thumbnail((1400, 1400))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=82)
            annotated = buf.getvalue()
        return {"detections": dets, "width": int(r.orig_shape[1]), "height": int(r.orig_shape[0]),
                "annotated": annotated, "speed": round(float(getattr(r, "speed", {}).get("inference", 0)), 1)}
    finally:
        tmp_in.unlink(missing_ok=True)


def predict_on_dataset_image(model_path: str, dataset_id: str, split: str, name: str,
                             conf: float = 0.25) -> dict:
    p = datasets.image_path(dataset_id, split, name)
    return predict(model_path, p.read_bytes(), conf=conf)


# ------------------------------------------------------------------ 导出

def export_model(model_path: str, fmt: str = "onnx", imgsz: int = 640,
                 half: bool = False, simplify: bool = True) -> dict:
    from ultralytics import YOLO
    model = YOLO(model_path)
    out = model.export(format=fmt, imgsz=imgsz, half=half, simplify=simplify)
    out_path = Path(str(out))
    if not out_path.is_absolute():
        out_path = (Path(model_path).parent / out_path).resolve()
    size = out_path.stat().st_size if out_path.exists() else 0
    return {"format": fmt, "path": str(out_path), "size": size,
            "exists": out_path.exists(), "name": out_path.name}


# ------------------------------------------------------------------ 演示数据集

def make_demo_dataset(name: str = "demo-shapes", n: int = 120, classes: list[str] | None = None) -> str:
    """生成合成数据集（彩色几何形状），用于快速跑通全流程。"""
    classes = classes or ["circle", "rectangle", "triangle"]
    ds_id = datasets.create_empty(name, classes, task="detect")
    train_img = datasets.images_dir(ds_id) / "train"
    train_lab = datasets.labels_dir(ds_id) / "train"
    val_img = datasets.images_dir(ds_id) / "val"
    val_lab = datasets.labels_dir(ds_id) / "val"

    rng = random.Random(7)
    colors = [(220, 60, 60), (60, 140, 230), (60, 190, 110), (235, 180, 50), (170, 90, 210)]

    def draw_shape(dr: ImageDraw.ImageDraw, kind: str, cx: float, cy: float, s: float, color: tuple) -> tuple:
        """返回归一化包围盒 (cx,cy,w,h)。"""
        if kind == "circle":
            dr.ellipse([cx - s, cy - s, cx + s, cy + s], fill=color, outline=(20, 20, 20), width=2)
            return (cx, cy, 2 * s, 2 * s)
        if kind == "rectangle":
            w, h = s * 1.6, s * 0.9
            dr.rectangle([cx - w, cy - h, cx + w, cy + h], fill=color, outline=(20, 20, 20), width=2)
            return (cx, cy, 2 * w, 2 * h)
        r = s * 1.15
        pts = [(cx, cy - r), (cx + r * 0.9, cy + r * 0.7), (cx - r * 0.9, cy + r * 0.7)]
        dr.polygon(pts, fill=color, outline=(20, 20, 20))
        return (cx, cy, r * 1.8, r * 1.85)

    def gen(idx: int, out_img: Path, out_lab: Path, size: int = 480) -> None:
        im = Image.new("RGB", (size, size), (rng.randint(232, 250),) * 3)
        dr = ImageDraw.Draw(im)
        n_obj = rng.randint(1, 3)
        lines: list[str] = []
        used: list[tuple] = []
        for _ in range(n_obj):
            cls = rng.randrange(len(classes))
            s = rng.randint(int(size * 0.07), int(size * 0.15))
            for _try in range(30):
                cx = rng.uniform(s + 8, size - s - 8)
                cy = rng.uniform(s + 8, size - s - 8)
                if all(abs(cx - ux) > (s + 26) or abs(cy - uy) > (s + 26) for ux, uy in used):
                    break
            used.append((cx, cy))
            ncx, ncy, nw, nh = draw_shape(dr, classes[cls], cx, cy, s, colors[cls % len(colors)])
            lines.append(f"{cls} {ncx/size:.6f} {ncy/size:.6f} {nw/size:.6f} {nh/size:.6f}")
        im.save(out_img, quality=90)
        out_lab.write_text("\n".join(lines), encoding="utf-8")

    for i in range(n):
        if i < int(n * 0.8):
            gen(i, train_img / f"img_{i:04d}.jpg", train_lab / f"img_{i:04d}.txt")
        else:
            gen(i, val_img / f"img_{i:04d}.jpg", val_lab / f"img_{i:04d}.txt")

    datasets.write_data_yaml(ds_id, classes)
    meta = datasets.read_meta(ds_id)
    meta["source"] = "demo"
    datasets.write_meta(ds_id, meta)
    return ds_id


# ------------------------------------------------------------------ 预训练模型仓库

PRETRAINED = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt"]

# GitHub 直连失败时的镜像回退（国内网络常见）
_MIRRORS = [
    "https://hf-mirror.com/Ultralytics/YOLO11/resolve/main/{file}",
    "https://hf-mirror.com/Ultralytics/YOLOv8/resolve/main/{file}",
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/{file}",
]


def _download_weights(weights: str, target: Path) -> bool:
    """依次尝试镜像源下载权重，成功返回 True。"""
    import urllib.request

    for tpl in _MIRRORS:
        url = tpl.format(file=weights)
        try:
            print(f"[weights] 尝试 {url}", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(target, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            if target.stat().st_size > 100_000:
                print(f"[weights] 成功: {url} ({target.stat().st_size} bytes)", flush=True)
                return True
            target.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            print(f"[weights] 失败: {e}", flush=True)
            target.unlink(missing_ok=True)
    return False


def ensure_pretrained(model_id: str) -> str:
    """确保预训练权重已下载到本地 models/pretrained/（GitHub 不通时自动走镜像）。"""
    weights = model_id if model_id.endswith(".pt") else f"{model_id}.pt"
    if weights.startswith("yolo11") and weights not in PRETRAINED:
        pass  # 允许任意 yolo11* 名称
    target_dir = config.MODELS_DIR / "pretrained"
    target_dir.mkdir(parents=True, exist_ok=True)
    local = target_dir / weights
    if local.exists() and local.stat().st_size > 100_000:
        return str(local)
    # 1) 优先 ultralytics 官方下载（会校验并缓存到 cwd）
    try:
        from ultralytics import YOLO
        YOLO(weights)
        for cand in [Path.cwd() / weights, config.BASE_DIR / weights]:
            if cand.exists():
                shutil.move(str(cand), local)
                return str(local)
    except Exception as e:  # noqa: BLE001
        print(f"[weights] ultralytics 官方下载失败: {e}", flush=True)
    # 2) 回退镜像
    if _download_weights(weights, local):
        return str(local)
    raise RuntimeError(f"无法获取 {weights}：请检查网络，或手动把 .pt 文件放入 data/models/pretrained/")


def pretrained_list() -> list[dict]:
    d = config.MODELS_DIR / "pretrained"
    out = []
    for w in PRETRAINED:
        p = d / w
        out.append({"id": w.replace(".pt", ""), "file": w,
                    "downloaded": p.exists(), "size": p.stat().st_size if p.exists() else 0})
    return out


def register_model_file(path: str, name: str, classes: list[str], source: str = "uploaded") -> str:
    return db.add_model(name=name, path=path, classes=classes, source=source,
                        size=Path(path).stat().st_size if Path(path).exists() else 0)


# ------------------------------------------------------------------ 本机模型导入

WEIGHT_EXTS = {".pt", ".pth", ".onnx", ".engine", ".torchscript", ".mlmodel", ".tflite"}
# label 文件放在磁盘时，ETag/last-modified 变化会导致缓存失效，统一用进程内字典即可


def inspect_weights(path: Path) -> dict:
    """读取本地权重文件的元信息：类别、任务类型、参数规模。

    支持 Ultralytics 的 .pt（含训练时写入的 names/task）以及无法解析的格式（返回空信息）。
    """
    info: dict = {"classes": [], "task": "detect", "nc": 0, "params": 0,
                  "format": path.suffix.lower().lstrip("."), "readable": False}
    if path.suffix.lower() != ".pt":
        # onnx/engine 等无法反查类别，交给用户手动填
        try:
            import onnx  # noqa: F401
            if path.suffix.lower() == ".onnx":
                info["format"] = "onnx"
        except ImportError:
            pass
        return info

    try:
        import torch
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        info["error"] = f"无法解析权重：{e}"
        return info

    try:
        if isinstance(ckpt, dict):
            model = ckpt.get("model") or ckpt.get("ema")
            names = ckpt.get("names")
            if names is None and model is not None:
                names = getattr(model, "names", None)
            if isinstance(names, dict):
                info["classes"] = [str(names[k]) for k in sorted(names)]
            elif isinstance(names, (list, tuple)):
                info["classes"] = [str(x) for x in names]
            task = ckpt.get("task") or getattr(model, "task", None)
            if task:
                info["task"] = str(task)
            info["nc"] = len(info["classes"])
            # 参数量
            try:
                m = model.model if hasattr(model, "model") else model
                info["params"] = int(sum(p.numel() for p in m.parameters()))
            except Exception:
                info["params"] = 0
            info["epochs"] = int(ckpt.get("epoch", -1)) + 1 if ckpt.get("epoch") is not None else 0
            info["readable"] = True
        elif hasattr(ckpt, "names"):
            info["classes"] = [str(x) for x in getattr(ckpt, "names", [])]
            info["task"] = str(getattr(ckpt, "task", "detect"))
            info["readable"] = True
    except Exception as e:  # noqa: BLE001
        info["error"] = f"元信息解析失败：{e}"
    return info
