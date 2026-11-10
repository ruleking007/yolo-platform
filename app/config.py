"""YOLO 训练平台 —— 全局路径与常量配置。"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent          # yolo-platform/
DATA_DIR = BASE_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
RUNS_DIR = DATA_DIR / "runs"
MODELS_DIR = DATA_DIR / "models"
TMP_DIR = DATA_DIR / "tmp"
DB_PATH = DATA_DIR / "platform.sqlite3"
WEB_DIR = BASE_DIR / "web"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 内置模型：本机 data/models/pretrained/ 已缓存权重，离线可用、CPU 可训练
MODEL_CHOICES = [
    {"id": "yolo11n", "label": "YOLO11n · nano（最快，CPU 推荐）"},
    {"id": "yolo11s", "label": "YOLO11s · small"},
    {"id": "yolo11m", "label": "YOLO11m · medium（CPU 上较慢）"},
    {"id": "yolov8n", "label": "YOLOv8n · 经典稳定"},
]

EXPORT_FORMATS = ["onnx", "torchscript", "engine", "openvino", "coreml", "tflite", "pb", "saved_model", "ncnn", "paddle"]


def ensure_dirs() -> None:
    for p in (DATA_DIR, DATASETS_DIR, RUNS_DIR, MODELS_DIR, TMP_DIR):
        p.mkdir(parents=True, exist_ok=True)


ensure_dirs()
