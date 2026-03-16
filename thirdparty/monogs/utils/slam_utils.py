# Copyright 2024 The MonoGS Authors.

# Licensed under the License issued by the MonoGS Authors
# available here: https://github.com/muskie82/MonoGS/blob/main/LICENSE.md

import torch
from thirdparty.gaussian_splatting.utils.loss_utils import ssim
import torch.nn.functional as F
import numpy as np

def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


# Not used, but kept for reference
def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False, rm_dynamic=False, mask=None):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint, rm_dynamic=rm_dynamic, mask=mask)
def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint, rm_dynamic=False, mask=None):
    # 获取原始图像并转存到GPU
    gt_image = viewpoint.original_image.cuda()
    # 提取图像尺寸信息
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)

    # 生成RGB有效区域掩码（过滤过暗区域）
    rgb_boundary_threshold = config["mapping"]["Training"]["rgb_boundary_threshold"]  # 从配置读取亮度阈值
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    # 结合梯度掩码（关注边缘区域）
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask

    # 动态物体掩码处理（当需要移除动态物体且非首帧时）
    if viewpoint.motion_mask is not None and rm_dynamic and viewpoint.uid > 0:
        rgb_pixel_mask = viewpoint.motion_mask.view(*mask_shape) * rgb_pixel_mask

    # 应用外部传入的额外掩码（如重投影掩码）
    if mask is not None:
        rgb_pixel_mask = mask.view(*mask_shape) * rgb_pixel_mask

    # 计算加权L1损失（不透明度*RGB差异）
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()  # 返回平均损失值

# Not used, but kept for reference
def get_loss_tracking_rgbd(
        config, image, depth, opacity, viewpoint, initialization=False, rm_dynamic=False, mask=None
):
    # 设置RGB与深度损失的混合权重（默认0.95:0.05）
    alpha = config["mapping"]["Training"]["alpha"] if "alpha" in config["mapping"]["Training"] else 0.95

    # 准备深度真值数据（转Tensor并匹配设备）
    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    #gt_depth=viewpoint.depth[None]
    # 生成深度有效区域掩码（过滤过近/过远点）
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)  # 近裁剪面0.01米
    depth_pixel_mask *= (gt_depth < 1000.).view(*depth.shape)  # 远裁剪面1000米
    # 基于不透明度的可靠区域掩码
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    # 计算RGB分量损失
    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint, rm_dynamic=rm_dynamic, mask=mask)
    # 合成最终深度掩码（深度有效区域∩高不透明度区域）
    depth_mask = depth_pixel_mask * opacity_mask

    # 动态物体掩码处理
    if viewpoint.motion_mask is not None and rm_dynamic and viewpoint.uid > 0:
        depth_mask = viewpoint.motion_mask.view(*depth.shape) * depth_mask

    # 应用外部传入的额外掩码
    if mask is not None:
        depth_mask = mask.view(*depth.shape) * depth_mask

    # 计算深度L1损失
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    # 返回加权总损失（RGB权重α，深度权重1-α）
    return (1 - alpha) * l1_rgb + alpha * l1_depth.mean()


# ---------------------------------------------------------------------------
# Mapping loss：统一为  w_rgb * L1_rgb + w_depth * L1_depth，掩码与权重均从 config 读取
# ---------------------------------------------------------------------------

def _get_mapping_loss_config(config):
    """从 config['Training'] 读取 mapping loss 权重与阈值，避免散落魔法数。"""
    t = config.get("Training", config)
    return {
        "alpha": t.get("alpha", 0.9),                    # RGB 权重，深度权重 = 1 - alpha
        "rgb_boundary_threshold": t.get("rgb_boundary_threshold", 0.01),
        "depth_min": t.get("depth_min", 0.01),
        "depth_max": t.get("depth_max", 10000.0),
        "dynamic_region_weight": t.get("dynamic_region_weight", 1.0),  # 动态区域 L1 权重倍数，1 表示不加重
    }


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False, alpha=None, rm_dynamic=False,
                     mask=None, dynamic=False, split=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    return get_loss_mapping_rgbd(
        config, image_ab, depth, viewpoint,
        alpha=alpha, rm_dynamic=rm_dynamic, mask=mask, dynamic=dynamic, split=split,
    )


def get_loss_mapping_rgb(config, image, depth, viewpoint):
    """纯 RGB L1，用于 monocular 模式。"""
    cfg = _get_mapping_loss_config(config)
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    th = cfg["rgb_boundary_threshold"]
    rgb_mask = (gt_image.sum(dim=0) > th).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_mask - gt_image * rgb_mask)
    n = (rgb_mask.sum() * 3).float().clamp(min=1e-8)
    return (l1_rgb * rgb_mask.unsqueeze(0)).sum() / n


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False, alpha=None, rm_dynamic=False,
                          mask=None, dynamic=False, split=False):
    """
    统一 mapping 重建损失：L = w_rgb * L1_rgb + w_depth * L1_depth。
    - 有效像素：rgb_boundary_threshold 过滤黑边，depth 在 [depth_min, depth_max]。
    - rm_dynamic=True 时用 motion_mask 只保留静态区域；mask 额外与有效区域交。
    - dynamic=True 时对动态区域误差乘 dynamic_region_weight（配置项，默认 1.0）。
    - split=True 时返回 (loss_static, loss_dynamic) 供外部加权。
    """
    cfg = _get_mapping_loss_config(config)
    w_rgb = alpha if alpha is not None else cfg["alpha"]
    w_depth = 1.0 - w_rgb
    th_rgb = cfg["rgb_boundary_threshold"]
    d_min, d_max = cfg["depth_min"], cfg["depth_max"]
    dyn_weight = cfg["dynamic_region_weight"]

    gt_image = viewpoint.original_image.cuda()
    if not isinstance(viewpoint.depth, torch.Tensor):
        viewpoint.depth = torch.from_numpy(np.asarray(viewpoint.depth)).float()
    gt_depth = viewpoint.depth.to(dtype=torch.float32, device=image.device)[None]

    # 有效区域掩码
    rgb_mask = (gt_image.sum(dim=0) > th_rgb).view(*depth.shape)
    depth_mask = (gt_depth > d_min).view(*depth.shape) & (gt_depth < d_max).view(*depth.shape)

    if viewpoint.motion_mask is not None and rm_dynamic:
        m = viewpoint.motion_mask.view(*depth.shape)
        rgb_mask = rgb_mask & m
        depth_mask = depth_mask & m
    if mask is not None and rm_dynamic:
        m = mask.view(*depth.shape)
        rgb_mask = rgb_mask & m
        depth_mask = depth_mask & m

    l1_rgb = torch.abs(image * rgb_mask - gt_image * rgb_mask)
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)

    if split and viewpoint.motion_mask is not None:
        motion = viewpoint.motion_mask.view(*depth.shape)
        sm = (rgb_mask & motion).float()
        dm = (depth_mask & motion).float()
        static_rgb = (l1_rgb * sm.unsqueeze(0)).sum() / (sm.sum() * 3 + 1e-8)
        static_depth = (l1_depth * dm).sum() / (dm.sum() + 1e-8)
        sm_d = (rgb_mask & ~motion).float()
        dm_d = (depth_mask & ~motion).float()
        dyn_rgb = (l1_rgb * sm_d.unsqueeze(0)).sum() / (sm_d.sum() * 3 + 1e-8)
        dyn_depth = (l1_depth * dm_d).sum() / (dm_d.sum() + 1e-8)
        loss_static = w_rgb * static_rgb + w_depth * static_depth
        loss_dynamic = w_rgb * dyn_rgb + w_depth * dyn_depth
        if dynamic:
            return loss_static, dyn_weight * loss_dynamic
        return loss_static, loss_dynamic

    # 动态区域加权：对非静态像素的误差乘 dyn_weight
    if dynamic and viewpoint.motion_mask is not None and dyn_weight != 1.0:
        non_static = ~viewpoint.motion_mask.view(*depth.shape)
        l1_rgb = l1_rgb * (1.0 + (dyn_weight - 1.0) * non_static.unsqueeze(0).float())
        l1_depth = l1_depth * (1.0 + (dyn_weight - 1.0) * non_static.float())

    n_rgb = rgb_mask.sum().float().clamp(min=1e-8)
    n_depth = depth_mask.sum().float().clamp(min=1e-8)
    loss_rgb = (l1_rgb * rgb_mask).sum() / n_rgb
    loss_depth = (l1_depth * depth_mask).sum() / n_depth
    return w_rgb * loss_rgb + w_depth * loss_depth


def depth_loss_dpt(pred_depth, gt_depth, weight=None):
    """
    :param pred_depth:  (H, W)
    :param gt_depth:    (H, W)
    :param weight:      (H, W)
    :return:            scalar
    """
    if isinstance(gt_depth, np.ndarray):
        gt_depth = torch.from_numpy(gt_depth).float().to(pred_depth.device)

    gt_depth = gt_depth.to(pred_depth.device)
    pred_depth=pred_depth.squeeze()
    t_pred = torch.median(pred_depth)
    s_pred = torch.mean(torch.abs(pred_depth - t_pred))

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))

    pred_depth_n = (pred_depth - t_pred) / s_pred
    gt_depth_n = (gt_depth - t_gt) / s_gt
    gt_depth_n = gt_depth_n.to(pred_depth_n.device)
    if weight is not None:
        loss = F.mse_loss(pred_depth_n, gt_depth_n, reduction='none')
        #print(f"[损失计算后] loss.device = {loss.device}")
        if isinstance(weight, np.ndarray):
            weight = torch.from_numpy(weight).float().to(pred_depth.device)
        weight=weight.float().to(pred_depth.device)
        #print(f"[损失计算后] weight.device = {weight.device}")
        loss = loss * weight
        loss = loss.sum() / (weight.sum() + 1e-8)
    else:
        loss = F.mse_loss(pred_depth_n, gt_depth_n)
    return loss

def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()
