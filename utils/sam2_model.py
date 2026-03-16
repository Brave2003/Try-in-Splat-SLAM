# utils/sam2_model.py
import os
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 相对 try 工程根目录；run 时需在 try 根目录或 PYTHONPATH 含 try；ckpt 统一放在 pretrained/
_TRY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SAM2_ROOT = os.path.join(_TRY_ROOT, "sam2")
_PRETRAINED_DIR = os.path.join(_TRY_ROOT, "pretrained")

_SAM2_PRETRAINED = {
    "sam2.1_hiera_tiny": {
        "cfg": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "ckpt": os.path.join(_PRETRAINED_DIR, "sam2.1_hiera_base_plus.pt"),
    },
    "sam2.1_hiera_small": {
        "cfg": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "ckpt": os.path.join(_PRETRAINED_DIR, "sam2.1_hiera_small.pt"),
    },
    "sam2.1_hiera_base_plus": {
        "cfg": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "ckpt": os.path.join(_PRETRAINED_DIR, "sam2.1_hiera_base_plus.pt"),
    },
}


def get_sam2_predictor(
    name: str = "sam2.1_hiera_base_plus",
    device: str = "cuda",
):
    info = _SAM2_PRETRAINED.get(name) or _SAM2_PRETRAINED["sam2.1_hiera_base_plus"]
    ckpt = info["ckpt"]
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing ckpt: {ckpt}")
    model = build_sam2(info["cfg"], ckpt)
    predictor = SAM2ImagePredictor(model.to(device))
    return predictor


def segment_image_with_points(
    predictor,
    image,
    points=None,
    labels=None,
    pos_mask=None,
    neg_mask=None,
    max_points_per_type=64,
):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    pts, lbs = [], []

    def sample_from_mask(mask, label):
        if mask is None:
            return
        mask = mask.astype(bool)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return
        idx = np.random.choice(len(xs), size=min(max_points_per_type, len(xs)), replace=False)
        for i in idx:
            pts.append([xs[i], ys[i]])
            lbs.append(label)

    sample_from_mask(pos_mask, 1)
    sample_from_mask(neg_mask, 0)

    if points is not None:
        points = np.asarray(points, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)
        pts.extend(points.tolist())
        lbs.extend(labels.tolist())

    if len(pts) == 0:
        return None, None

    pts = np.asarray(pts, dtype=np.float32)
    lbs = np.asarray(lbs, dtype=np.int32)

    predictor.set_image(image)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks, scores, _ = predictor.predict(
            point_coords=pts[None, ...],
            point_labels=lbs[None, ...],
            multimask_output=False,
        )

    masks = masks[0]
    scores = scores[0]
    return masks, scores
