"""图片规范化：EXIF 方向纠正、格式统一为 JPEG、去重哈希、最长边压缩。

客户直接丢进来的图片往往很杂：手机 HEIC/PNG/长宽不一/带旋转信息/文件名重复/
超大原图。统一在这里处理，标注与训练阶段就只面对干净一致的 JPEG。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

MAX_SIDE_DEFAULT = 1920     # 标注和训练都用不到比这更大的分辨率
JPEG_QUALITY = 92
MAX_FILE_BYTES = 40 * 1024 * 1024


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path, size: int = 16) -> str:
    """感知哈希：缩放到 (size+1)×size 灰度后比较左右相邻像素，对缩放/重编码鲁棒。

    size=16 得到 256 位，比常见的 8×8(64 位) 更能分辨"局部新增物体"这类差异。
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L").resize((size + 1, size), Image.LANCZOS)
        px = list(im.getdata())
    bits = []
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits.append("1" if left > right else "0")
    return "".join(bits)


def aHash(path: Path, size: int = 16) -> str:
    """平均哈希：与整体均值比较，对亮度整体偏移不敏感。"""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L").resize((size, size), Image.LANCZOS)
        px = list(im.getdata())
    avg = sum(px) / len(px)
    return "".join("1" if v >= avg else "0" for v in px)


def hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def similarity(a: str, b: str) -> float:
    """0~1，1 表示完全一致。"""
    if len(a) != len(b) or not a:
        return 0.0
    return 1.0 - hamming(a, b) / len(a)


def is_visual_duplicate(ph_a: str, ph_b: str, ah_a: str = "", ah_b: str = "") -> bool:
    """判定视觉重复：dHash（256 位）+ aHash 双重校验。

    实测：同一张图重新压缩/缩放 → dHash 相似度 1.000；
          同底图但新增物体 → 约 0.941（256 位差 15 位）。
    因此 dHash 阈值取 0.97（允许 8/256 位差异），足以容忍 JPEG 重压与缩放，
    又不会把"新增了目标物体"的图误删。
    """
    if not ph_a or not ph_b:
        return False
    if similarity(ph_a, ph_b) < 0.97:
        return False
    if ah_a and ah_b and similarity(ah_a, ah_b) < 0.95:
        return False
    return True


def normalize(source: Path, dest: Path, max_side: int = MAX_SIDE_DEFAULT) -> dict:
    """把任意可解码图片写成干净 JPEG。返回尺寸信息；失败抛异常由调用方记录。"""
    with Image.open(source) as im:
        im.load()
        orig_w, orig_h = im.size
        im = ImageOps.exif_transpose(im)          # 手机照片的旋转信息
        if im.mode in ("RGBA", "LA", "P"):
            # 透明图铺白底，避免转 RGB 后变成黑块
            rgba = im.convert("RGBA")
            canvas = Image.new("RGB", rgba.size, (255, 255, 255))
            canvas.paste(rgba, mask=rgba.split()[-1])
            im = canvas
        elif im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        if im.mode == "L":
            im = im.convert("RGB")

        w, h = im.size
        if max_side and max(w, h) > max_side:
            im.thumbnail((max_side, max_side), Image.LANCZOS)
            w, h = im.size

        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)

    return {"width": w, "height": h, "orig_width": orig_w, "orig_height": orig_h,
            "bytes": dest.stat().st_size}


def is_decodable(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def clean_filename(name: str) -> str:
    """保留可读文件名，剔除路径分隔符与危险字符。"""
    name = (name or "image").replace("\\", "/").split("/")[-1]
    bad = '<>:"|?*\x00\n\r\t'
    for ch in bad:
        name = name.replace(ch, "_")
    stem = Path(name).stem.strip(" .") or "image"
    return stem[:80] + ".jpg"
