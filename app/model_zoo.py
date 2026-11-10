"""模型结构改进（视频方向二：网络架构 / 注意力机制 / 多尺度融合 / 轻量化）。

原理：ultralytics 的 parse_model 按 YAML 逐行搭网络，通道/深度由 scales 字典自动缩放。
本模块以官方 yolo11 / yolov8 结构为底，按勾选的改进项增删替换层，生成结构 YAML
（写入 data/models/zoo/），再用对应预训练 .pt 做部分权重迁移——即论文里最常见的
「改进 YOLO + 迁移学习」实验范式。结构不同的新层（CBAM/PSA/P2 分支）自动随机初始化。

改进项（全部为 ultralytics 内置模块，无第三方依赖）：
  cbam   主干末端加 CBAM 通道+空间双注意力
  psa    注意力升级：yolo11 的 C2PSA→PSA 单层版；yolov8 在 SPPF 后补 PSA
  p2     增加 P2/4 小目标检测层（4 层 PAN 融合，显存/耗时约 ×2~3）
  ghost  主干 Conv→GhostConv、基础块→C3Ghost（GhostNet 轻量化）
"""
from __future__ import annotations

import re

from . import config

_INJECTS = ("CBAM",)  # CBAM 存在于 nn.modules 但不在 tasks 的导入列表


def _inject_modules() -> None:
    import ultralytics.nn.tasks as T
    from ultralytics.nn import modules as M

    for name in _INJECTS:
        if not hasattr(T, name) and hasattr(M, name):
            setattr(T, name, getattr(M, name))


ARCH_OPTIONS = [
    {"key": "cbam", "name": "CBAM 注意力",
     "desc": "通道+空间双注意力，接在主干末端。参数几乎不变，突出目标抑制背景，小数据集常见收益。"},
    {"key": "psa", "name": "PSA 空间注意力",
     "desc": "YOLOv10 同款金字塔空间注意力。yolo11 用它替换 C2PSA，yolov8 在 SPPF 后新增。"},
    {"key": "p2", "name": "P2 小目标检测头",
     "desc": "增加 P2/4 高分辨率层专治小目标漏检。代价大：显存与每轮耗时约 ×2~3，CPU 慎用。"},
    {"key": "ghost", "name": "GhostConv 轻量化",
     "desc": "GhostNet 思路，主干卷积换 GhostConv、块换 C3Ghost，参数/计算量下降，精度略降。"},
]
ARCH_KEYS = {o["key"] for o in ARCH_OPTIONS}

IOU_LOSS_OPTIONS = [
    {"key": "", "name": "CIoU（官方默认）", "desc": "IoU − 中心距 − 宽高比一致性三项权衡。"},
    {"key": "wiou", "name": "Wise-IoU v1",
     "desc": "动态非单调聚焦：离群低质量样本自动降权，回归更稳（Wisebounding 2021）。"},
    {"key": "siou", "name": "SIoU",
     "desc": "角度+距离+形状三重代价，收敛更快，对中心点偏移不敏感（SIoU 2022）。"},
]
IOU_LOSS_KEYS = {o["key"] for o in IOU_LOSS_OPTIONS}

_SCALES = {  # 官方 scales：[depth, width, max_channels]
    ("yolo11", "n"): (0.50, 0.25, 1024), ("yolo11", "s"): (0.50, 0.50, 1024),
    ("yolo11", "m"): (0.50, 1.00, 512),
    ("yolov8", "n"): (0.33, 0.25, 1024), ("yolov8", "s"): (0.33, 0.50, 1024),
    ("yolov8", "m"): (0.67, 0.75, 768),
}

# 官方 backbone 基础块 (重复数, 通道目标, shortcut, e)；head 块 (重复数, 通道目标, shortcut)
_FAMILIES = {
    "yolo11": {
        "block": "C3k2", "light": "C3Ghost",
        "bb": [(2, 256, False, 0.25), (2, 512, False, 0.25), (2, 512, True, None), (2, 1024, True, None)],
        "head": [(2, 512, False), (2, 256, False), (2, 512, False), (2, 1024, True)],
        "c2psa": True,
    },
    "yolov8": {
        "block": "C2f", "light": "C3Ghost",
        "bb": [(3, 128, True, None), (6, 256, True, None), (6, 512, True, None), (3, 1024, True, None)],
        "head": [(3, 512, None), (3, 256, None), (3, 512, None), (3, 1024, None)],
        "c2psa": False,
    },
}


def family_of(base: str) -> str | None:
    m = re.match(r"(yolo11|yolov8)", str(base) or "")
    return m.group(1) if m else None


def scale_of(base: str) -> str:
    m = re.search(r"([nsm])$", str(base) or "")
    return m.group(1) if m else "n"


def supported(base: str) -> bool:
    return (family_of(base), scale_of(base)) in _SCALES


def _md(c: float) -> int:
    """make_divisible(c, 8)：与 ultralytics 同式（8 的向上取整）。"""
    import math

    return int(8 * math.ceil(round(float(c) / 8.0, 6)))


def _blk_args(mod: str, c: int, shortcut, e) -> str:
    """按模块习惯拼参数列表（与官方各家族 yaml 一致）。"""
    parts = [str(c)]
    if shortcut is not None:
        parts.append("True" if shortcut else "False")
    if e is not None and mod == "C3k2":
        parts.append(str(e))
    return "[" + ", ".join(parts) + "]"


def build_rows(fam: str, a: set[str]) -> tuple[list, int]:
    """生成 (from, n, module, args) 行列表 + backbone 结束行号。

    与官方 yaml 对齐：通道目标写官方值，缩放交给 parse_model 的 scales；
    仅 CBAM（不在 base_modules，不自动缩放）需要写死缩放后通道。
    """
    F = _FAMILIES[fam]
    ghost = "ghost" in a
    conv = "GhostConv" if ghost else "Conv"
    blk = F["light"] if ghost else F["block"]
    rows: list[tuple] = []

    def add(f, n, mod, args):
        rows.append((f, n, mod, args))
        return len(rows) - 1

    # ---- backbone（0 stem, 1 P2/4, 3 P3/8, 5 P4/16, 7 P5/32）----
    add(-1, 1, "Conv", "[64, 3, 2]")
    add(-1, 1, conv, "[128, 3, 2]")
    b1 = add(-1, F["bb"][0][0], blk, _blk_args(blk, *F["bb"][0][1:]))
    add(-1, 1, conv, "[256, 3, 2]")
    b2 = add(-1, F["bb"][1][0], blk, _blk_args(blk, *F["bb"][1][1:]))
    add(-1, 1, conv, "[512, 3, 2]")
    b3 = add(-1, F["bb"][2][0], blk, _blk_args(blk, *F["bb"][2][1:]))
    add(-1, 1, conv, "[1024, 3, 2]")
    b4 = add(-1, F["bb"][3][0], blk, _blk_args(blk, *F["bb"][3][1:]))
    sppf = add(-1, 1, "SPPF", "[1024, 5]")
    if F["c2psa"]:
        last_bb = add(-1, 1 if "psa" in a else 2, "PSA" if "psa" in a else "C2PSA", "[1024]")
    elif "psa" in a:
        last_bb = add(-1, 1, "PSA", "[1024]")
    else:
        last_bb = sppf
    if "cbam" in a:
        last_bb = add(-1, 1, "CBAM", "[1024, 7]")  # build_yaml 里替换为缩放后通道
    bb_end = last_bb

    # ---- head：自上而下 P4 ← P3 ← (P2) ← 下采样回 P4/P5 ----
    h4a, h3a, h4b, h5a = F["head"]

    up1 = add(-1, 1, "nn.Upsample", '[None, 2, "nearest"]')
    add([up1, b3], 1, "Concat", "[1]")
    n4 = add(-1, h4a[0], blk, _blk_args(blk, h4a[1], h4a[2], None))
    up2 = add(-1, 1, "nn.Upsample", '[None, 2, "nearest"]')
    add([up2, b2], 1, "Concat", "[1]")
    n3 = add(-1, h3a[0], blk, _blk_args(blk, h3a[1], h3a[2], None))

    small_out, p3_out = [], n3
    if "p2" in a:
        up3 = add(-1, 1, "nn.Upsample", '[None, 2, "nearest"]')
        add([up3, b1], 1, "Concat", "[1]")
        n2 = add(-1, h3a[0], blk, _blk_args(blk, 128, h3a[2], None))
        small_out = [n2]
        d0 = add(-1, 1, conv, "[128, 3, 2]")
        add([d0, n3], 1, "Concat", "[1]")
        p3_out = add(-1, h3a[0], blk, _blk_args(blk, h3a[1], h3a[2], None))

    d1 = add(-1, 1, conv, "[256, 3, 2]")
    add([d1, n4], 1, "Concat", "[1]")
    n4b = add(-1, h4b[0], blk, _blk_args(blk, h4b[1], h4b[2], None))
    d2 = add(-1, 1, conv, "[512, 3, 2]")
    add([d2, last_bb], 1, "Concat", "[1]")
    n5 = add(-1, h5a[0], blk, _blk_args(blk, h5a[1], h5a[2], None))

    add(small_out + [p3_out, n4b, n5], 1, "Detect", "[nc]")
    return rows, bb_end


def build_yaml(base: str, arch: list[str], nc: int) -> str:
    """生成结构 YAML 文本（含 scales 字典；文件名决定 scale）。"""
    _inject_modules()
    fam = family_of(base)
    if not fam:
        raise ValueError("结构改进目前仅支持内置 yolo11 / yolov8 系列基础模型")
    scale = scale_of(base)
    if (fam, scale) not in _SCALES:
        raise ValueError(f"基础模型 {base} 不在结构改进支持范围（n/s/m）")
    depth, width, max_ch = _SCALES[(fam, scale)]
    a = set(arch) & ARCH_KEYS
    rows, bb_end = build_rows(fam, a)

    # CBAM 不在 base_modules → 参数不自动缩放，写死缩放后的通道
    c5 = _md(min(1024, max_ch) * width)
    out_rows = [(f, n, m, (f"[{c5}, 7]" if m == "CBAM" else args)) for f, n, m, args in rows]

    def fmt(f):
        return str(f) if isinstance(f, int) else "[" + ", ".join(str(x) for x in f) + "]"

    lines = [
        f"# 平台生成：{base} + {'+'.join(k for k in ('p2','ghost','cbam','psa') if k in a) or '原版'}",
        "# 通道/深度由 scales 自动缩放（与官方 yaml 相同机制）",
        f"nc: {nc}",
        "scales:",
        f"  {scale}: [{depth}, {width}, {max_ch}]",
        "backbone:",
    ]
    for i, (f, n, m, args) in enumerate(out_rows):
        if i == bb_end + 1:
            lines.append("head:")
        lines.append(f"  - [{fmt(f)}, {n}, {m}, {args}]")
    return "\n".join(lines) + "\n"


def arch_slug(base: str, arch: list[str]) -> str:
    parts = [str(base).removesuffix(".pt").removesuffix(".yaml")]
    parts += [k for k in ("p2", "ghost", "cbam", "psa") if k in (set(arch) & ARCH_KEYS)]
    return "-".join(parts)


def ensure_zoo_yaml(base: str, arch: list[str], nc: int) -> str:
    """生成（或复用）结构 YAML 文件，返回绝对路径。"""
    zoo = config.MODELS_DIR / "zoo"
    zoo.mkdir(parents=True, exist_ok=True)
    path = zoo / f"{arch_slug(base, arch)}.yaml"
    text = build_yaml(base, arch, nc)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")
    return str(path)
