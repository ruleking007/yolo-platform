"""数据集管理：上传/解压、YOLO 结构校验、自动划分、标注读写、缩略图。

数据集目录约定（data/datasets/<dataset_id>/）：
    images/train  images/val          图片
    labels/train  labels/val          YOLO txt 标注（与图片同名）
    data.yaml                          nc / names / path / train / val
    meta.json                          平台侧元信息
"""
from __future__ import annotations

import io
import json
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image

from . import config

Image.MAX_IMAGE_PIXELS = None

_SAFE = re.compile(r"[^0-9A-Za-z_\u4e00-\u9fff.-]+")


def slugify(name: str, fallback: str = "dataset") -> str:
    s = _SAFE.sub("_", (name or "").strip()).strip("._")
    return (s[:60] or fallback)


def ds_dir(dataset_id: str) -> Path:
    return config.DATASETS_DIR / dataset_id


def images_dir(dataset_id: str) -> Path:
    return ds_dir(dataset_id) / "images"


def labels_dir(dataset_id: str) -> Path:
    return ds_dir(dataset_id) / "labels"


def meta_path(dataset_id: str) -> Path:
    return ds_dir(dataset_id) / "meta.json"


def yaml_path(dataset_id: str) -> Path:
    return ds_dir(dataset_id) / "data.yaml"


def safe_join(root: Path, rel: str) -> Path:
    """把相对路径安全拼到 root 下，阻止越权访问。"""
    rel = (rel or "").replace("\\", "/").lstrip("/")
    p = (root / rel).resolve()
    root_r = root.resolve()
    if p != root_r and root_r not in p.parents:
        raise ValueError(f"非法路径: {rel}")
    return p


# ---------------------------------------------------------------- meta

def read_meta(dataset_id: str) -> dict:
    p = meta_path(dataset_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def write_meta(dataset_id: str, meta: dict) -> None:
    ds_dir(dataset_id).mkdir(parents=True, exist_ok=True)
    meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
    meta_path(dataset_id).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def list_datasets() -> list[dict]:
    out = []
    for d in sorted(config.DATASETS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        meta = read_meta(d.name)
        stat = scan_stats(d.name)
        out.append({
            "id": d.name,
            "name": meta.get("name", d.name),
            "classes": meta.get("classes", []),
            "task": meta.get("task", "detect"),
            "created_at": meta.get("created_at"),
            "archived": bool(meta.get("archived")),
            **stat,
        })
    return out


def scan_stats(dataset_id: str) -> dict:
    """扫描图片/标注数量与已标注数。"""
    stats = {"n_images": 0, "n_labeled": 0, "n_train": 0, "n_val": 0, "boxes": 0}
    img_root = images_dir(dataset_id)
    if not img_root.exists():
        return stats
    for sub in ("train", "val"):
        d = img_root / sub
        if not d.exists():
            continue
        files = [f for f in d.iterdir() if f.suffix.lower() in config.IMAGE_EXTS]
        stats[f"n_{sub}"] = len(files)
        stats["n_images"] += len(files)
        lab = labels_dir(dataset_id) / sub
        for f in files:
            lp = lab / (f.stem + ".txt")
            if lp.exists() and lp.stat().st_size > 0:
                stats["n_labeled"] += 1
                try:
                    stats["boxes"] += sum(1 for line in lp.read_text(encoding="utf-8").splitlines() if line.strip())
                except Exception:
                    pass
    return stats


# ---------------------------------------------------------------- 纯图片导入（客户零门槛流程）

def create_from_images(name: str, classes: list[str] | None = None,
                       task: str = "detect", max_side: int = 1920) -> str:
    """先建数据集骨架，图片随后用 ingest_images 批量灌入（边传边建，不阻塞）。"""
    classes = [c.strip() for c in (classes or ["object"]) if str(c).strip()] or ["object"]
    return create_empty(name or "dataset", classes, task)


def ingest_images(dataset_id: str, files: list[tuple[str, Path]],
                  split: str = "train", max_side: int = 1920,
                  dedupe: bool = True) -> dict:
    """把整批原始图片规范化后放进数据集。

    files: [(原始文件名, 已落盘的临时文件路径)]
    返回 {added, skipped_duplicate, skipped_invalid, errors}
    """
    from . import images as img

    dst_dir = images_dir(dataset_id) / split
    dst_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir(dataset_id) / split).mkdir(parents=True, exist_ok=True)

    seen = _existing_hashes(dataset_id)          # {路径: sha1}
    seen_ph = _load_hash_index(dataset_id).get("phash", {})   # {路径: dHash}
    seen_ah = _load_hash_index(dataset_id).get("ahash", {})   # {路径: aHash}
    added = dup = invalid = 0
    dups: list[str] = []
    errors: list[str] = []
    added_stems: list[str] = []

    for orig_name, src in files:
        try:
            raw = src.read_bytes()
        except Exception as e:  # noqa: BLE001
            invalid += 1
            errors.append(f"{orig_name}: 读取失败 {e}")
            continue
        if not raw:
            invalid += 1
            errors.append(f"{orig_name}: 空文件")
            continue
        if len(raw) > img.MAX_FILE_BYTES:
            invalid += 1
            errors.append(f"{orig_name}: 超过 {img.MAX_FILE_BYTES // 1024 // 1024}MB")
            continue

        base = img.clean_filename(orig_name)
        dest = _unique(dst_dir, base)
        try:
            img.normalize(src, dest, max_side=max_side)
        except Exception as e:  # noqa: BLE001
            invalid += 1
            errors.append(f"{orig_name}: 无法解码（{type(e).__name__}）")
            continue

        # 两级去重：先精确内容哈希，再感知哈希（识别重新压缩/缩放的同一张图）
        digest = img.sha1_file(dest)
        phash = img.dhash(dest)
        ahash = img.aHash(dest)
        hit = None
        if dedupe:
            for key, h in seen.items():
                if h == digest:
                    hit = key
                    break
            if hit is None:
                for key, ph in seen_ph.items():
                    if img.is_visual_duplicate(ph, phash, seen_ah.get(key, ""), ahash):
                        hit = key
                        break
        if hit is not None:
            dest.unlink(missing_ok=True)
            dup += 1
            dups.append(f"{orig_name} ≈ {hit}")
            continue

        key = f"{split}/{dest.name}"
        seen[key] = digest
        seen_ph[key] = phash
        seen_ah[key] = ahash
        added += 1
        added_stems.append(dest.stem)

    _save_hash_index(dataset_id, seen, seen_ph, seen_ah)
    return {"added": added, "skipped_duplicate": dup, "skipped_invalid": invalid,
            "duplicates": dups[:30], "errors": errors[:50], "stems": added_stems}


_HASH_INDEX = "_image_hashes.json"


def _hash_index_path(dataset_id: str) -> Path:
    return ds_dir(dataset_id) / _HASH_INDEX


def _save_hash_index(dataset_id: str, sha: dict, ph: dict, ah: dict | None = None) -> None:
    p = _hash_index_path(dataset_id)
    try:
        p.write_text(json.dumps({"sha1": sha, "phash": ph, "ahash": ah or {}},
                                ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _load_hash_index(dataset_id: str) -> dict:
    p = _hash_index_path(dataset_id)
    if p.exists():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "sha1" in raw:
                return {"sha1": raw.get("sha1") or {},
                        "phash": raw.get("phash") or {},
                        "ahash": raw.get("ahash") or {}}
            return {"sha1": raw if isinstance(raw, dict) else {}, "phash": {}, "ahash": {}}
        except Exception:
            pass
    return {"sha1": {}, "phash": {}, "ahash": {}}


def _existing_hashes(dataset_id: str) -> dict[str, str]:
    """已有图片的内容哈希 {相对路径: sha1}，同时补全感知哈希。"""
    from . import images as img
    idx = _load_hash_index(dataset_id)
    sha: dict = dict(idx["sha1"])
    ph: dict = dict(idx["phash"])
    ah: dict = dict(idx["ahash"])

    root = images_dir(dataset_id)
    if not root.exists():
        return sha

    dirty = False
    for f in root.rglob("*"):
        if not (f.is_file() and f.suffix.lower() in config.IMAGE_EXTS):
            continue
        key = f.relative_to(root).as_posix()      # train/xxx.jpg，避免跨 split 同名冲突
        if not sha.get(key):
            try:
                sha[key] = img.sha1_file(f)
                dirty = True
            except Exception:
                continue
        if not ph.get(key):
            try:
                ph[key] = img.dhash(f)
                ah[key] = img.aHash(f)
                dirty = True
            except Exception:
                pass
    if dirty:
        _save_hash_index(dataset_id, sha, ph, ah)
    return sha


def finalize_dataset(dataset_id: str, classes: list[str] | None = None,
                     split_ratio: float = 0.8, task: str | None = None,
                     name: str | None = None) -> dict:
    """导入收尾：类别落定、自动划分 train/val、写 data.yaml 与 meta。"""
    meta = read_meta(dataset_id)
    if classes:
        meta["classes"] = [c.strip() for c in classes if str(c).strip()]
    if task:
        meta["task"] = task
    if name:
        meta["name"] = slugify(name, dataset_id)
    meta.setdefault("classes", ["object"])
    meta.setdefault("task", "detect")
    meta.setdefault("name", dataset_id)
    meta.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    write_meta(dataset_id, meta)

    stats = scan_stats(dataset_id)
    if stats["n_val"] == 0 and stats["n_train"] > 1:
        split_existing(dataset_id, split_ratio)
    write_data_yaml(dataset_id, meta["classes"])
    return scan_stats(dataset_id)


# ---------------------------------------------------------------- 创建 / 导入

def create_empty(name: str, classes: list[str], task: str = "detect") -> str:
    from . import db
    ds_id = db.new_id("d_")
    d = ds_dir(ds_id)
    for sub in ("train", "val"):
        (d / "images" / sub).mkdir(parents=True, exist_ok=True)
        (d / "labels" / sub).mkdir(parents=True, exist_ok=True)
    write_data_yaml(ds_id, classes)
    write_meta(ds_id, {
        "id": ds_id, "name": slugify(name, "dataset"), "classes": classes, "task": task,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    return ds_id


def write_data_yaml(dataset_id: str, classes: list[str]) -> None:
    data = {
        "path": str(ds_dir(dataset_id).resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes,
    }
    yaml_path(dataset_id).write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _read_classes_from_data_yaml(dataset_id: str) -> list[str]:
    p = yaml_path(dataset_id)
    if not p.exists():
        return []
    try:
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    names = d.get("names") or []
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


def import_upload(filename: str, stream, name: str | None = None,
                  task: str = "detect", split_ratio: float = 0.8) -> str:
    """导入上传内容。支持 zip（YOLO 结构 / 图片包）与单张图片。"""
    from . import db
    ds_id = db.new_id("d_")
    d = ds_dir(ds_id)
    d.mkdir(parents=True, exist_ok=True)
    staging = config.TMP_DIR / db.new_id("up_")
    staging.mkdir(parents=True, exist_ok=True)

    suffix = Path(filename or "upload").suffix.lower()
    saved = staging / (Path(filename or "upload").stem or "upload")
    with open(saved, "wb") as fh:
        shutil.copyfileobj(stream, fh)

    classes: list[str] = []
    try:
        if suffix == ".zip":
            classes = _import_zip(saved, ds_id)
        elif suffix in config.IMAGE_EXTS:
            _import_single_image(saved, ds_id)
        else:
            raise ValueError(f"不支持的文件类型：{suffix or '(无扩展名)'}。请上传 .zip 或图片。")

        if not classes:
            classes = _infer_classes_from_labels(ds_id)
        if not classes:
            classes = ["object"]
            _rewrite_all_labels_to_class0(ds_id)

        # 已带 train/val 结构的数据集不要二次打乱；只有 val 为空时才自动划分
        if scan_stats(ds_id)["n_val"] == 0:
            split_existing(ds_id, split_ratio)
        write_data_yaml(ds_id, classes)
        write_meta(ds_id, {
            "id": ds_id, "name": slugify(name or Path(filename).stem, "dataset"),
            "classes": classes, "task": task,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": "upload",
        })
        return ds_id
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _import_zip(zip_path: Path, dataset_id: str) -> list[str]:
    """解压 zip 并归一化到 images/<split> + labels/<split>。返回 data.yaml 里的类别（若有）。"""
    extract = config.TMP_DIR / f"x_{zip_path.stem}"
    shutil.rmtree(extract, ignore_errors=True)
    extract.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        _safe_extract(zf, extract)

    classes = _find_data_yaml_classes(extract)
    consumed = _copy_structured(extract, dataset_id)   # 已符合 YOLO 结构的部分

    # 剩余散图 -> images/train（稍后统一划分）
    tmp_train = images_dir(dataset_id) / "train"
    tmp_labels = labels_dir(dataset_id) / "train"
    tmp_train.mkdir(parents=True, exist_ok=True)
    tmp_labels.mkdir(parents=True, exist_ok=True)
    loose = 0
    for p in extract.rglob("*"):
        if not (p.is_file() and p.suffix.lower() in config.IMAGE_EXTS):
            continue
        if p.resolve() in consumed:
            continue
        dest = _unique(tmp_train, p.name)
        shutil.copy2(p, dest)
        loose += 1
        lp = p.with_suffix(".txt")
        if not lp.exists():
            lp = _find_sibling_label(p, extract)
        if lp and lp.exists():
            shutil.copy2(lp, tmp_labels / (dest.stem + ".txt"))
            sj = lp.parent / (lp.stem + ".shapes.json")   # 边车与 txt 同目录
            if sj.exists():
                shutil.copy2(sj, tmp_labels / (dest.stem + ".shapes.json"))
    shutil.rmtree(extract, ignore_errors=True)
    if not consumed and loose == 0:
        raise ValueError("压缩包里没有找到图片或 YOLO 标注文件。")
    return classes


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    dest_r = dest.resolve()
    for member in zf.namelist():
        target = (dest / member).resolve()
        if dest_r not in target.parents and target != dest_r:
            raise ValueError(f"压缩包内含非法路径：{member}")
    zf.extractall(dest)


def _find_data_yaml_classes(root: Path) -> list[str]:
    for p in list(root.rglob("data.yaml")) + list(root.rglob("data.yml")):
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        names = d.get("names")
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names)]
        if isinstance(names, list):
            return [str(x) for x in names]
    return []


def _copy_structured(root: Path, dataset_id: str) -> set[Path]:
    """复制 <任意>/images/{train,val,test} 结构，返回已消费图片的路径集合。"""
    consumed: set[Path] = set()
    img_bases = [d for d in root.rglob("images")
                 if d.is_dir() and any((d / s).is_dir() for s in ("train", "val", "test"))]
    for img_base in img_bases:
        lab_base = img_base.parent / "labels"
        for split in ("train", "val", "test"):
            src = img_base / split
            if not src.is_dir():
                continue
            dst_split = "val" if split in ("val", "test") else "train"
            dst_img = images_dir(dataset_id) / dst_split
            dst_lab = labels_dir(dataset_id) / dst_split
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lab.mkdir(parents=True, exist_ok=True)
            for p in sorted(src.rglob("*")):
                if not (p.is_file() and p.suffix.lower() in config.IMAGE_EXTS):
                    continue
                dest = _unique(dst_img, p.name)
                shutil.copy2(p, dest)
                consumed.add(p.resolve())
                lp = lab_base / split / (p.stem + ".txt")
                if lp.exists():
                    shutil.copy2(lp, dst_lab / (dest.stem + ".txt"))
                sj = lab_base / split / (p.stem + ".shapes.json")
                if sj.exists():
                    shutil.copy2(sj, dst_lab / (dest.stem + ".shapes.json"))
    return consumed


def _find_sibling_label(image: Path, root: Path) -> Path | None:
    for cand in (image.with_suffix(".txt"),):
        if cand.exists():
            return cand
    for cand in root.rglob(f"**/{image.stem}.txt"):
        if cand.is_file():
            return cand
    return None


def _import_single_image(image: Path, dataset_id: str) -> None:
    dst = images_dir(dataset_id) / "train"
    dst.mkdir(parents=True, exist_ok=True)
    (labels_dir(dataset_id) / "train").mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, _unique(dst, image.name))


def _infer_classes_from_labels(dataset_id: str) -> list[str]:
    """无 data.yaml 时，从类别名文件夹推断（images/train/cat/*.jpg）。"""
    names: list[str] = []
    for split in ("train", "val"):
        d = images_dir(dataset_id) / split
        if not d.exists():
            continue
        for sub in d.iterdir():
            if sub.is_dir() and sub.name not in names:
                names.append(sub.name)
    return names


def _rewrite_all_labels_to_class0(dataset_id: str) -> None:
    for sub in ("train", "val"):
        d = labels_dir(dataset_id) / sub
        if not d.exists():
            continue
        for lp in d.glob("*.txt"):
            lines = []
            for line in _read_text(lp).splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    parts[0] = "0"
                    lines.append(" ".join(parts))
            lp.write_text("\n".join(lines), encoding="utf-8")


def _unique(dst_dir: Path, filename: str) -> Path:
    p = dst_dir / filename
    if not p.exists():
        return p
    stem, suf = p.stem, p.suffix
    i = 1
    while (dst_dir / f"{stem}_{i}{suf}").exists():
        i += 1
    return dst_dir / f"{stem}_{i}{suf}"


# ---------------------------------------------------------------- 划分

def split_existing(dataset_id: str, ratio: float = 0.8, seed: int = 0) -> dict:
    """把 images/train 里的图片按比例重新划分到 train/val（标注同步移动）。"""
    import random
    train_img = images_dir(dataset_id) / "train"
    train_lab = labels_dir(dataset_id) / "train"
    if not train_img.exists():
        return scan_stats(dataset_id)

    files = sorted([f for f in train_img.iterdir() if f.suffix.lower() in config.IMAGE_EXTS])
    if not files:
        return scan_stats(dataset_id)

    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = max(1, int(round(len(files) * (1 - ratio)))) if len(files) > 1 else 0
    val_set = set(files[:n_val])

    (images_dir(dataset_id) / "val").mkdir(parents=True, exist_ok=True)
    (labels_dir(dataset_id) / "val").mkdir(parents=True, exist_ok=True)
    (train_lab.parent).mkdir(parents=True, exist_ok=True)

    for f in val_set:
        shutil.move(str(f), _unique(images_dir(dataset_id) / "val", f.name))
        for ext in (".txt", ".shapes.json"):
            lp = train_lab / (f.stem + ext)
            if lp.exists():
                shutil.move(str(lp), _unique(labels_dir(dataset_id) / "val", lp.name))
    # 图片跨 split 移动后，路径键的哈希索引失效，重建一次
    _hash_index_path(dataset_id).unlink(missing_ok=True)
    _existing_hashes(dataset_id)
    return scan_stats(dataset_id)

# ---------------------------------------------------------------- 标注读写

def _read_text(p: Path) -> str:
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return ""


def shapes_path(dataset_id: str, split: str, stem: str) -> Path:
    """形状边车文件：保存原始形状（circle/ellipse/polygon），txt 只是训练格式投影。"""
    return labels_dir(dataset_id) / split / (stem + ".shapes.json")


def _bbox_of(b: dict) -> tuple[float, float, float, float]:
    """任意形状 -> 归一化外接框 (cx, cy, w, h)。"""
    sh = b.get("shape", "box")
    if sh == "polygon" and b.get("points"):
        pts = b["points"]
        xs, ys = pts[0::2], pts[1::2]
        if not xs:
            return (0.0, 0.0, 0.0, 0.0)
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    if sh in ("circle", "ellipse"):
        return (b["cx"], b["cy"], 2 * b["rx"], 2 * b["ry"])
    return (b.get("cx", 0), b.get("cy", 0), b.get("w", 0), b.get("h", 0))


def _ellipse_points(cx: float, cy: float, rx: float, ry: float, n: int = 36) -> list[float]:
    import math
    out: list[float] = []
    for i in range(n):
        a = 2 * math.pi * i / n
        out.append(round(cx + rx * math.cos(a), 6))
        out.append(round(cy + ry * math.sin(a), 6))
    return out


def _polygon_of(b: dict) -> list[float] | None:
    """任意形状 -> 归一化多边形点列（segment 训练用）。"""
    sh = b.get("shape", "box")
    if sh == "polygon" and b.get("points"):
        return [round(float(x), 6) for x in b["points"]]
    if sh in ("circle", "ellipse"):
        return _ellipse_points(b["cx"], b["cy"], b["rx"], b["ry"])
    if sh == "box":
        cx, cy, w, h = b["cx"], b["cy"], b["w"], b["h"]
        return [round(cx - w / 2, 6), round(cy - h / 2, 6),
                round(cx + w / 2, 6), round(cy - h / 2, 6),
                round(cx + w / 2, 6), round(cy + h / 2, 6),
                round(cx - w / 2, 6), round(cy + h / 2, 6)]
    return None


def list_images(dataset_id: str, split: str | None = None, q: str = "",
                labeled: str | None = None, offset: int = 0, limit: int = 60) -> dict:
    root = images_dir(dataset_id)
    if not root.exists():
        return {"total": 0, "items": [], "splits": []}
    splits = [s for s in ("train", "val") if (root / s).exists()]
    if split:
        splits = [s for s in splits if s == split]
    items: list[dict] = []
    for s in splits:
        for f in sorted((root / s).iterdir()):
            if not (f.is_file() and f.suffix.lower() in config.IMAGE_EXTS):
                continue
            if q and q.lower() not in f.name.lower():
                continue
            boxes = read_label(dataset_id, s, f.stem)
            if labeled == "yes" and not boxes:
                continue
            if labeled == "no" and boxes:
                continue
            items.append({"split": s, "name": f.name, "stem": f.stem,
                          "n_boxes": len(boxes), "labeled": bool(boxes)})
    total = len(items)
    return {"total": total, "splits": splits, "items": items[offset: offset + limit]}


def read_label(dataset_id: str, split: str, stem: str) -> list[dict]:
    """兼容旧接口：返回带 shape 字段的形状列表。"""
    return read_shapes(dataset_id, split, stem)


def read_shapes(dataset_id: str, split: str, stem: str) -> list[dict]:
    """读取形状（优先边车 json，缺失时从 txt 推断）。

    统一结构：
      box:     {shape:'box', cls, cx, cy, w, h}
      circle:  {shape:'circle', cls, cx, cy, rx, ry}
      ellipse: {shape:'ellipse', cls, cx, cy, rx, ry}
      polygon: {shape:'polygon', cls, points:[x,y,...]}
    """
    sj = shapes_path(dataset_id, split, stem)
    if sj.exists():
        try:
            data = json.loads(_read_text(sj) or "[]")
            if isinstance(data, list):
                return [_normalize_shape(b) for b in data if isinstance(b, dict)]
        except Exception:
            pass

    p = labels_dir(dataset_id) / split / (stem + ".txt")
    if not p.exists():
        return []
    out = []
    for line in _read_text(p).splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        if len(vals) >= 6 and len(vals) % 2 == 0:
            out.append(_poly_or_box(cls, vals))
        else:
            out.append({"shape": "box", "cls": cls,
                        "cx": vals[0], "cy": vals[1], "w": vals[2], "h": vals[3]})
    return out


def _poly_or_box(cls: int, vals: list[float]) -> dict:
    """4 点且轴对齐的多边形还原为 box（老 segment 数据集里的矩形能正常拖角点编辑）。"""
    if len(vals) == 8:
        xs, ys = vals[0::2], vals[1::2]
        if len({round(v, 6) for v in xs}) == 2 and len({round(v, 6) for v in ys}) == 2:
            return {"shape": "box", "cls": cls,
                    "cx": (min(xs) + max(xs)) / 2, "cy": (min(ys) + max(ys)) / 2,
                    "w": max(xs) - min(xs), "h": max(ys) - min(ys)}
    return {"shape": "polygon", "cls": cls, "points": vals}


def _normalize_shape(b: dict) -> dict:
    sh = b.get("shape") or ("polygon" if b.get("points") else "box")
    if sh == "polygon":
        return {"shape": "polygon", "cls": int(b.get("cls", 0)),
                "points": [float(x) for x in (b.get("points") or [])]}
    if sh in ("circle", "ellipse"):
        return {"shape": sh, "cls": int(b.get("cls", 0)),
                "cx": float(b["cx"]), "cy": float(b["cy"]),
                "rx": float(b.get("rx", 0)), "ry": float(b.get("ry", 0))}
    return {"shape": "box", "cls": int(b.get("cls", 0)),
            "cx": float(b.get("cx", 0)), "cy": float(b.get("cy", 0)),
            "w": float(b.get("w", 0)), "h": float(b.get("h", 0))}


def write_label(dataset_id: str, split: str, stem: str, boxes: list[dict],
                task: str = "detect") -> None:
    """写形状：原始形状存边车 json；txt 按任务类型投影（detect=外接框，segment=多边形）。"""
    d = labels_dir(dataset_id) / split
    d.mkdir(parents=True, exist_ok=True)

    norm = [_normalize_shape(b) for b in boxes]

    lines = []
    for b in norm:
        cls = b["cls"]
        if task == "segment":
            pts = _polygon_of(b)
            if pts:
                lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in pts))
        else:
            cx, cy, w, h = _bbox_of(b)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    (d / (stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")

    # 边车：只要出现非 box 形状就写；全是 box 时删除边车保持目录干净
    sj = shapes_path(dataset_id, split, stem)
    if any(b["shape"] != "box" for b in norm):
        sj.write_text(json.dumps(norm, ensure_ascii=False), encoding="utf-8")
    else:
        sj.unlink(missing_ok=True)

    from . import db
    db.touch_annotation(dataset_id, f"{split}/{stem}")


def image_path(dataset_id: str, split: str, name: str) -> Path:
    return safe_join(images_dir(dataset_id), f"{split}/{name}")


def thumbnail(dataset_id: str, split: str, name: str, size: int = 256) -> bytes:
    p = image_path(dataset_id, split, name)
    with Image.open(p) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=72)
        return buf.getvalue()


def full_sized(dataset_id: str, split: str, name: str, max_side: int = 1600) -> bytes:
    p = image_path(dataset_id, split, name)
    with Image.open(p) as im:
        im = im.convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


def delete_image(dataset_id: str, split: str, name: str) -> None:
    p = image_path(dataset_id, split, name)
    if p.exists():
        p.unlink()
    stem = Path(name).stem
    lp = labels_dir(dataset_id) / split / (stem + ".txt")
    if lp.exists():
        lp.unlink()
    shapes_path(dataset_id, split, stem).unlink(missing_ok=True)


def add_images(dataset_id: str, files: list[tuple[str, bytes]], split: str = "train") -> int:
    dst = images_dir(dataset_id) / split
    dst.mkdir(parents=True, exist_ok=True)
    (labels_dir(dataset_id) / split).mkdir(parents=True, exist_ok=True)
    n = 0
    for name, data in files:
        if Path(name).suffix.lower() not in config.IMAGE_EXTS:
            continue
        p = _unique(dst, Path(name).name)
        p.write_bytes(data)
        n += 1
    return n


def set_classes(dataset_id: str, classes: list[str]) -> None:
    meta = read_meta(dataset_id)
    meta["classes"] = classes
    meta["name"] = meta.get("name", dataset_id)
    write_meta(dataset_id, meta)
    write_data_yaml(dataset_id, classes)


def rename_dataset(dataset_id: str, name: str) -> None:
    meta = read_meta(dataset_id)
    meta["name"] = slugify(name, dataset_id)
    write_meta(dataset_id, meta)


def delete_dataset(dataset_id: str) -> None:
    shutil.rmtree(ds_dir(dataset_id), ignore_errors=True)


def export_zip(dataset_id: str) -> Path:
    """打包数据集（images + labels + data.yaml）。"""
    out = config.TMP_DIR / f"{dataset_id}.zip"
    base = ds_dir(dataset_id)
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in base.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(base).as_posix())
    return out


def get_classes(dataset_id: str) -> list[str]:
    meta = read_meta(dataset_id)
    return meta.get("classes") or _read_classes_from_data_yaml(dataset_id)


def get_task(dataset_id: str) -> str:
    return read_meta(dataset_id).get("task", "detect")
