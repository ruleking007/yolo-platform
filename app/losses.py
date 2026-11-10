"""Loss 函数改进：Wise-IoU v1 / SIoU，可替换 ultralytics 默认的 CIoU 回归损失。

对应视频「训练策略优化 → 损失函数改进」方向。

原理速记：
  CIoU  = IoU − ρ²/c² − α·v          （对低质量框梯度不稳）
  WIoU  = IoU − (ρ²/c²)^γ, γ=α·ρ²/c² （非单调动态聚焦：离群样本自动降权）
  SIoU  = IoU − s·Λ·(Ωw+Ωh)          （引入角度代价 + 距离代价 + 形状代价，收敛更快）

用法（train_worker 内）：
    from app.losses import patch_bbox_iou
    patch_bbox_iou("wiou")   # 之后 ultralytics BboxLoss 内部调用即生效
"""
from __future__ import annotations

import math

import torch

_EPS = 1e-7


def _xyxy(box1: torch.Tensor, box2: torch.Tensor, xywh: bool):
    """统一转成 (x1,y1,x2,y2) 分量。"""
    if xywh:
        (x1_1, y1_1, w1, h1), (x1_2, y1_2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1 = [x1_1, y1_1, x1_1 + w1, y1_1 + h1]
        b2 = [x1_2, y1_2, x1_2 + w2, y1_2 + h2]
    else:
        b1 = box1.unbind(-1)
        b2 = box2.unbind(-1)
    return b1, b2


def _iou_terms(box1: torch.Tensor, box2: torch.Tensor, xywh: bool):
    """返回 (iou, w1,h1,w2,h2, xc1,yc1,xc2,yc2, enc_w,enc_h)。"""
    (x1_1, y1_1, x2_1, y2_1), (x1_2, y1_2, x2_2, y2_2) = _xyxy(box1, box2, xywh)
    w1, h1 = (x2_1 - x1_1).clamp(min=0), (y2_1 - y1_1).clamp(min=0)
    w2, h2 = (x2_2 - x1_2).clamp(min=0), (y2_2 - y1_2).clamp(min=0)
    inter_w = (torch.min(x2_1, x2_2) - torch.max(x1_1, x1_2)).clamp(min=0)
    inter_h = (torch.min(y2_1, y2_2) - torch.max(y1_1, y1_2)).clamp(min=0)
    inter = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - inter
    iou = inter / (union + _EPS)
    xc1, yc1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
    xc2, yc2 = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
    enc_w = torch.max(x2_1, x2_2) - torch.min(x1_1, x1_2)
    enc_h = torch.max(y2_1, y2_2) - torch.min(y1_1, y1_2)
    return iou, w1, h1, w2, h2, xc1, yc1, xc2, yc2, enc_w, enc_h


def bbox_iou_wiou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, eps: float = _EPS):
    """Wise-IoU v1（对齐官方实现）：IoU − d1·d2·C/(ρ²+C)。

    d1=x^γ, d2=x^(1/γ), x=ρ²/c², γ=α·x（α 随宽高比偏差自适应）。
    好框 x→0 时 d2→0 惩罚消失；离群框 x→1 时乘积→1 且系数≈1/2——
    对离低质量样本梯度非单调，自动降权（动态离群点测量）。
    """
    iou, w1, h1, w2, h2, xc1, yc1, xc2, yc2, enc_w, enc_h = _iou_terms(box1, box2, xywh)
    rho2 = (xc1 - xc2) ** 2 + (yc1 - yc2) ** 2
    c2 = enc_w ** 2 + enc_h ** 2 + eps
    x = rho2 / c2
    # 宽高比偏差 → 形状一致性度量 sin_v ∈ [0,1]，α ∈ [0.5, 1]
    sin_v = 1 - 2 * torch.abs(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) / math.pi
    with torch.no_grad():
        alpha = 0.5 / (1 - 0.5 * sin_v + eps)
        gamma = (alpha * x).clamp(min=1e-7)
    d1 = torch.pow(x, gamma)
    d2 = torch.pow(x, 1.0 / gamma)
    return iou - d1 * d2 * c2 / (rho2 + c2)


def bbox_iou_siou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True,
                  beta: float = 4.0, eps: float = _EPS):
    """SIoU：IoU − s·Λ·(Ωw+Ωh)，角度+距离+形状三重代价，降低对中心点偏移的敏感度。"""
    iou, w1, h1, w2, h2, xc1, yc1, xc2, yc2, enc_w, enc_h = _iou_terms(box1, box2, xywh)
    dx, dy = xc1 - xc2, yc1 - yc2
    rho2 = dx ** 2 + dy ** 2
    sigma = torch.sqrt(rho2) + eps
    c = torch.maximum(torch.maximum(enc_w, enc_h), sigma)  # 外接对角约束 c ≥ σ
    c2 = c ** 2 + eps
    # 角度代价 Λ ∈ [0,1]：中心连线与坐标轴夹角越接近 45° 惩罚越小
    lam = 1 - 2 * torch.sin(torch.arcsin((dy.abs() / sigma).clamp(max=1.0)) - math.pi / 4) ** 2
    # 距离代价 s ∈ [0,2]
    s = 2 - (torch.exp(-dx ** 2 / c2) + torch.exp(-dy ** 2 / c2))
    # 形状代价 Ω：以 GT 宽高为基准
    omega_w = 1 - torch.tanh((w1 - w2).abs() / (w2 + eps)) ** beta
    omega_h = 1 - torch.tanh((h1 - h2).abs() / (h2 + eps)) ** beta
    return iou - s * lam * (omega_w + omega_h)


_IOU_FUNCS = {"wiou": bbox_iou_wiou, "siou": bbox_iou_siou}


def patch_bbox_iou(kind: str) -> bool:
    """把 ultralytics.utils.loss 内部的 bbox_iou 换成指定变体（仅拦截 CIoU 调用点）。

    BboxLoss.forward 以 bbox_iou(..., CIoU=True) 调用，替换后回归损失即生效；
    其余调用（如验证 IoU）保持原实现。返回是否成功打补丁。
    """
    if kind not in _IOU_FUNCS:
        return False
    import ultralytics.utils.loss as L

    if getattr(L.bbox_iou, "_platform_patched", False):
        return True  # 已在补丁状态（worker 进程内只会打一次）
    orig = L.bbox_iou
    fn = _IOU_FUNCS[kind]

    def patched(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=_EPS):
        if CIoU:
            return fn(box1, box2, xywh=xywh, eps=eps)
        return orig(box1, box2, xywh=xywh, GIoU=GIoU, DIoU=DIoU, CIoU=CIoU, eps=eps)

    patched._platform_patched = True
    patched._kind = kind
    L.bbox_iou = patched
    return True
