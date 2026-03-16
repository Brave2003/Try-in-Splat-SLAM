#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import random
import sys
from datetime import datetime

import numpy as np
import torch


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def PILtoTorch2(pil_image):
    # resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(pil_image)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """
    # def helper(step):
    #     if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
    #         # Disable this parameter
    #         return 0.0
    #     if lr_delay_steps > 0:
    #         # A kind of reverse cosine decay.
    #         delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
    #             0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
    #         )
    #     else:
    #         delay_rate = 1.0
    #     t = np.clip(step / max_steps, 0, 1)
    #     log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    #     return delay_rate * log_lerp

    return helper
    # return helper(lr_init=lr_init, lr_final=lr_final,
    #               lr_delay_steps=lr_delay_steps, lr_delay_mult=lr_delay_mult, max_steps=max_steps)


def helper(
    step, lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
        # Disable this parameter
        return 0.0
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
        )
    else:
        delay_rate = 1.0
    t = np.clip(step / max_steps, 0, 1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return delay_rate * log_lerp
# def get_expon_lr_func(
#         lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
# ):
#     """
#     Copied from Plenoxels
#
#     Continuous learning rate decay function. Adapted from JaxNeRF
#     The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
#     is log-linearly interpolated elsewhere (equivalent to exponential decay).
#     If lr_delay_steps>0 then the learning rate will be scaled by some smooth
#     function of lr_delay_mult, such that the initial learning rate is
#     lr_init*lr_delay_mult at the beginning of optimization but will be eased back
#     to the normal learning rate when steps>lr_delay_steps.
#     :param conf: config subtree 'lr' or similar
#     :param max_steps: int, the number of steps during optimization.
#     :return HoF which takes step as input
#     """
#
#     def helper(step):
#         if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
#             # Disable this parameter
#             return 0.0
#         if lr_delay_steps > 0:
#             # A kind of reverse cosine decay.
#             delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
#                 0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
#             )
#         else:
#             delay_rate = 1.0
#         t = np.clip(step / max_steps, 0, 1)
#         log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
#         return delay_rate * log_lerp
#
#     return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty
def get_normals(z, camera_metadata):
    pixels = camera_metadata.get_pixels()
    y = (pixels[..., 1] - camera_metadata.principal_point_y) / camera_metadata.scale_factor_y
    x = (
        pixels[..., 0] - camera_metadata.principal_point_x - y * camera_metadata.skew
    ) / camera_metadata.scale_factor_x
    viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    viewdirs = torch.from_numpy(viewdirs).to(z.device)

    coords = viewdirs[None] * z[..., None]
    coords = coords.permute(0, 3, 1, 2)

    dxdu = coords[..., 0, :, 1:] - coords[..., 0, :, :-1]
    dydu = coords[..., 1, :, 1:] - coords[..., 1, :, :-1]
    dzdu = coords[..., 2, :, 1:] - coords[..., 2, :, :-1]
    dxdv = coords[..., 0, 1:, :] - coords[..., 0, :-1, :]
    dydv = coords[..., 1, 1:, :] - coords[..., 1, :-1, :]
    dzdv = coords[..., 2, 1:, :] - coords[..., 2, :-1, :]

    dxdu = torch.nn.functional.pad(dxdu, (0, 1), mode="replicate")
    dydu = torch.nn.functional.pad(dydu, (0, 1), mode="replicate")
    dzdu = torch.nn.functional.pad(dzdu, (0, 1), mode="replicate")

    dxdv = torch.cat([dxdv, dxdv[..., -1:, :]], dim=-2)
    dydv = torch.cat([dydv, dydv[..., -1:, :]], dim=-2)
    dzdv = torch.cat([dzdv, dzdv[..., -1:, :]], dim=-2)

    n_x = dydv * dzdu - dydu * dzdv
    n_y = dzdv * dxdu - dzdu * dxdv
    n_z = dxdv * dydu - dxdu * dydv

    pred_normal = torch.stack([n_x, n_y, n_z], dim=-3)
    pred_normal = torch.nn.functional.normalize(pred_normal, dim=-3)
    return pred_normal
def get_gs_mask(s_image_tensor, gt_image_tensor, s_depth_tensor, depth_tensor, CVD):
    B, C, H, W = s_image_tensor.shape

    # Color based
    gs_error = torch.mean(torch.abs(s_image_tensor - gt_image_tensor), 1, True)
    gs_mask_c = error_to_prob(gs_error.detach())

    # Depth based
    gs_mask_d = error_to_prob(torch.mean(torch.abs(s_depth_tensor - depth_tensor), 1, True).detach())
    norm_disp = 1 / (CVD + 1e-7)
    norm_disp = (norm_disp + F.max_pool2d(-norm_disp, kernel_size=(H, W))) / (
        F.max_pool2d(norm_disp, kernel_size=(H, W)) + F.max_pool2d(-norm_disp, kernel_size=(H, W))
    )
    gs_mask_d = 1 - norm_disp * (1 - gs_mask_d)

    return gs_mask_c.detach(), gs_mask_d.detach()


def get_pixels(image_size_x, image_size_y, use_center=None):
    """Return the pixel at center or corner."""
    xx, yy = np.meshgrid(
        np.arange(image_size_x, dtype=np.float32),
        np.arange(image_size_y, dtype=np.float32),
    )
    offset = 0.5 if use_center else 0
    return np.stack([xx, yy], axis=-1) + offset


def error_to_prob(error, mask=None, mean_prob=0.5):
    if mask is None:
        mean_err = torch.mean(error, dim=(3, 2, 1)) + 1e-7
    else:
        mean_err = torch.sum(mask * error, dim=(3, 2)) / (torch.sum(mask, dim=(3, 2)) + 1e-7) + 1e-7
    prob = mean_prob * (error / mean_err.view(error.shape[0], 1, 1, 1))
    prob[prob > 1] = 1
    prob = 1 - prob
    return prob
def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def build_rotation_matrix_from_normal(normals):
    """
    Build rotation matrices so that the third column (local z-axis) aligns with the given world-space normals.
    Used to initialize Gaussian orientations from depth-derived normals (shortest axis = normal).

    Args:
        normals: (N, 3) tensor, unit normals in world space.

    Returns:
        R: (N, 3, 3) rotation matrices with R[:, :, 2] = normals (and degenerate -> identity).
    """
    N = normals.shape[0]
    device = normals.device
    n = torch.nn.functional.normalize(normals.float(), dim=1, eps=1e-8)
    # Tangent frame: pick a not parallel to n, then t1 = cross(a,n), t2 = cross(n,t1)
    a = torch.tensor([[1, 0, 0]], dtype=n.dtype, device=device).expand(N, 3).clone()
    mask = (n[:, 0].abs() > 0.9)
    a[mask] = torch.tensor([[0, 1, 0]], dtype=n.dtype, device=device)
    t1 = torch.cross(a, n, dim=1)
    t1_norm = t1.norm(dim=1, keepdim=True).clamp(min=1e-8)
    t1 = t1 / t1_norm
    t2 = torch.cross(n, t1, dim=1)
    t2 = torch.nn.functional.normalize(t2, dim=1, eps=1e-8)
    # Degenerate: n too small -> identity third column [0,0,1]
    valid = (normals.norm(dim=1) > 1e-4)
    R = torch.zeros((N, 3, 3), device=device, dtype=n.dtype)
    R[:, :, 0] = t1
    R[:, :, 1] = t2
    R[:, :, 2] = n
    R[~valid, :, 2] = 0.0
    R[~valid, 2, 2] = 1.0
    R[~valid, 0, 0] = 1.0
    R[~valid, 1, 1] = 1.0
    return R


def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion.

    Args:
    R (Tensor): A batch of 3x3 rotation matrices.

    Returns:
    Tensor: A batch of quaternions.
    """
    # Preallocate quaternion tensor
    q = torch.zeros((R.size(0), 4), device=R.device)

    # Calculate each element of the quaternion
    q[:, 0] = torch.sqrt(torch.max(torch.tensor(0.0, device=R.device), 1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2])) / 2
    q[:, 1] = torch.sqrt(torch.max(torch.tensor(0.0, device=R.device), 1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2])) / 2
    q[:, 2] = torch.sqrt(torch.max(torch.tensor(0.0, device=R.device), 1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2])) / 2
    q[:, 3] = torch.sqrt(torch.max(torch.tensor(0.0, device=R.device), 1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2])) / 2

    # Determine the correct signs
    q[:, 1] *= torch.sign(q[:, 1] * (R[:, 2, 1] - R[:, 1, 2]))
    q[:, 2] *= torch.sign(q[:, 2] * (R[:, 0, 2] - R[:, 2, 0]))
    q[:, 3] *= torch.sign(q[:, 3] * (R[:, 1, 0] - R[:, 0, 1]))

    return q

def quaternion_multiply(q1, q2):
    # Extract components
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    # Compute the product
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    
    return torch.stack((w, x, y, z), dim=-1)

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(
                                str(datetime.now().strftime("%d/%m %H:%M:%S"))
                            ),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    # sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
