# utils/sam2_infer.py
# -*- coding: utf-8 -*-

import os
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ---------------------------------------------------------
# 路径配置（try 工程根目录）
# ---------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

SAM2_CFG = os.path.join(PROJECT_ROOT, "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_b+.yaml")
# ckpt 统一放在 pretrained/（与 scripts/download_pretrained.sh 一致）
SAM2_CKPT = os.path.join(PROJECT_ROOT, "pretrained", "sam2.1_hiera_base_plus.pt")

_SAM2_PREDICTOR_CACHE = None
_SAM2_DEVICE = None


def load_sam2_predictor(device: str = "cuda") -> SAM2ImagePredictor:
    """
    返回一个全局复用的 SAM2ImagePredictor 实例。

    用法：
        from utils.sam2_infer import load_sam2_predictor
        predictor = load_sam2_predictor()
        predictor.set_image(rgb_np)
        masks, scores, logits = predictor.predict(...)
    """
    global _SAM2_PREDICTOR_CACHE, _SAM2_DEVICE

    if _SAM2_PREDICTOR_CACHE is not None and _SAM2_DEVICE == device:
        return _SAM2_PREDICTOR_CACHE

    if device == "cuda" and not torch.cuda.is_available():
        print("[SAM2] CUDA not available, fallback to CPU")
        device = "cpu"

    print(f"[SAM2] build_sam2 predictor: cfg={SAM2_CFG}, ckpt={SAM2_CKPT}, device={device}")

    model = build_sam2(config=SAM2_CFG, checkpoint=SAM2_CKPT, device=device)
    model.eval()

    predictor = SAM2ImagePredictor(model)

    _SAM2_PREDICTOR_CACHE = predictor
    _SAM2_DEVICE = device

    return predictor
