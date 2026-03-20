# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

# from encodings.punycode import selective_find
from PIL import Image
import cv2
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn

# from jsonschema.benchmarks.const_vs_enum import invalid
from tqdm import tqdm
import time
from colorama import Fore, Style
from multiprocessing.connection import Connection
from munch import munchify
from src.utils.datasets import BaseDataset
from src.utils.datasets import get_dataset, load_mono_depth
from src.utils.common import as_intrinsics_matrix, setup_seed
import matplotlib.pyplot as plt
from src.utils.Printer import Printer, FontColor
from thirdparty.glorie_slam.depth_video import DepthVideo
from thirdparty.gaussian_splatting.gaussian_renderer import render, render_flow
from thirdparty.gaussian_splatting.utils.general_utils import (
    rotation_matrix_to_quaternion,
    quaternion_multiply,
)
from thirdparty.gaussian_splatting.utils.loss_utils import l1_loss
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from thirdparty.gaussian_splatting.utils.graphics_utils import (
    getProjectionMatrix2,
    getWorld2View2,
)
from thirdparty.monogs.utils.pose_utils import update_pose
from thirdparty.monogs.utils.slam_utils import (
    get_loss_mapping,
    get_median_depth,
    depth_loss_dpt,
)
from thirdparty.monogs.utils.camera_utils import Camera

# 动态掩码改由前端保存图片，后端通过 dataset 读盘获得，不再在此导入
# from utils.dynamic_mask_detector import maybe_update_motion_masks
# from utils.eval_utils import  save_gaussians
from argparse import ArgumentParser
from arguments import ModelHiddenParams


def vis_render_process(
    gaussians,
    pipeline_params,
    background,
    viewpoint,
    cur_frame_idx,
    save_dir,
    out_dir="map",
    mask=None,
    dynamic=False,
    save_debug=False,
):
    """save_debug: 由配置 save_debug_visualization 控制，为 True 时才渲染并保存调试图"""
    if not save_debug:
        return
    with torch.no_grad():
        if dynamic:
            time_input = gaussians.deform.deform.expand_time(viewpoint.fid)
            d_values = gaussians.deform.step(
                gaussians.get_dygs_xyz.detach(),
                time_input,
                iteration=0,
                feature=None,
                motion_mask=gaussians.motion_mask,
                camera_center=viewpoint.camera_center,
                time_interval=gaussians.time_interval,
            )
            dxyz = d_values["d_xyz"]
            d_rot, d_scale = d_values["d_rotation"], d_values["d_scaling"]
            # print("scale: ", d_scale)
        else:
            dxyz, d_rot, d_scale = 0, 0, 0
        render_pkg = render(viewpoint, gaussians, pipeline_params, background)

        viz_im = torch.clip(render_pkg["render"].permute(1, 2, 0).detach().cpu(), 0, 1)

        h, w, _ = viz_im.shape
        fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
        cax = ax.imshow(viz_im)
        ax.axis("off")

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        os.makedirs(save_dir, exist_ok=True)
        process_dir = os.path.join(save_dir, out_dir)
        os.makedirs(process_dir, exist_ok=True)
        save_path = os.path.join(process_dir, f"{cur_frame_idx}.png")
        viz_im_np = np.array(viz_im)
        cv2.imwrite(save_path, viz_im_np)
        plt.savefig(save_path)
        plt.close()
        return


def _normal_to_rgb_01(normal_chw):
    """法向 (3,H,W) 用标准映射 [-1,1] -> [0,1] 得到可视化 RGB，与 depth_to_normal 约定一致。勿用 sigmoid。"""
    n = normal_chw.float()
    if n.dim() == 2:
        n = n.unsqueeze(0)
    rgb = (n.clamp(-1.0, 1.0) + 1.0) * 0.5
    return rgb  # (3,H,W) in [0,1]


def merge_hparams(args, config):
    params = ["ModelHiddenParams"]
    for param in params:
        if param in config.keys():
            for key, value in config[param].items():
                if hasattr(args, key):
                    setattr(args, key, value)
    return args


class Mapper(object):
    """
    Mapper thread.

    """

    # --------------- 1. 配置与初始化 ---------------
    def __init__(self, slam, pipe: Connection):
        # 步骤1：种子与配置
        setup_seed(slam.cfg["setup_seed"])
        torch.autograd.set_detect_anomaly(True)
        self.config = slam.cfg
        self.printer: Printer = slam.printer
        if self.config["only_tracking"]:
            return
        self.pipe = pipe
        self.verbose = slam.verbose
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.dtype = torch.float32
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = True
        self.keyframe_optimizers = None
        self.dystart = self.config["mapping"]["Training"].get("dystart", 11)
        self.video: DepthVideo = slam.video
        self.monocular = not self.initialized
        # 步骤2：模型/优化/管道参数
        model_params = munchify(self.config["mapping"]["model_params"])
        opt_params = munchify(self.config["mapping"]["opt_params"])
        pipeline_params = munchify(self.config["mapping"]["pipeline_params"])
        self.use_spherical_harmonics = self.config["mapping"]["Training"][
            "spherical_harmonics"
        ]
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )
        self.dynamic_model = self.config["mapping"]["model_params"]["dynamic_model"]
        parser = ArgumentParser(description="Training script parameters")
        hp = merge_hparams(ModelHiddenParams(parser), self.config["mapping"])
        self.sc_params = hp
        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0
        self.gaussians = GaussianModel(
            model_params.sh_degree,
            config=self.config,
            args=hp,
            init_deform=self.dynamic_model,
        )
        self.gaussians.init_lr(6.0)
        self.st_predicted = {}
        self.list = []
        self.first_d = []
        self.new_scale_alignFrame0 = dict()
        static_msk = np.ones((384, 512), dtype=bool)
        self.gaussians.training_setup(opt_params)
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.longer = self.config["mapping"]["long"]
        self.cameras_extent = 6.0
        self.dyratio = 0
        self.shift = self.config["mapping"]["shift"]
        self.s = self.config["mapping"]["s"]
        self.set_hyperparams()
        self.device = torch.device(self.config["device"])
        self.static_msk = torch.from_numpy(static_msk).to(self.device)
        self.frame_reader = get_dataset(self.config, device=self.device)
        if self.dynamic_model:
            self.gaussians.deform.train_setting(hp)
            self.gaussians.time_interval = 1 / len(self.frame_reader)

    def set_pipe(self, pipe):
        self.pipe = pipe

    def set_hyperparams(self):
        """步骤：从配置读取映射超参（初始化/稠密化/窗口/保存路径等）。"""
        c = self.config["mapping"]["Training"]
        self.gt_camera = c["gt_camera"]
        self.init_itr_num = c["init_itr_num"]
        self.init_gaussian_update = c["init_gaussian_update"]
        self.init_gaussian_reset = c["init_gaussian_reset"]
        self.init_gaussian_th = c["init_gaussian_th"]
        self.init_gaussian_extent = self.cameras_extent * c["init_gaussian_extent"]
        self.mapping_itr_num = c["mapping_itr_num"]
        self.gaussian_update_every = c["gaussian_update_every"]
        self.gaussian_update_offset = c["gaussian_update_offset"]
        self.gaussian_th = c["gaussian_th"]
        self.gaussian_extent = self.cameras_extent * c["gaussian_extent"]
        self.gaussian_reset = c["gaussian_reset"]
        self.size_threshold = c["size_threshold"]
        self.window_size = c["window_size"]
        self.flattening_loss_weight = c.get("flattening_loss_weight", 0.1)
        # 扁平化与法向相关权重：
        # - scale_loss_weight：PGSR 风格的 min(sx,sy,sz) 扁平约束（默认 100，与 PGSR 一致）
        # - flattening_loss_weight：法向先验损失权重（默认 5，可按需要调大/调小）
        self.scale_loss_weight = c.get("scale_loss_weight", 100.0)
        # rotation_lr：若提供，则在 mapping / refinement 阶段覆盖 rotation param_group 的 lr，加快法向相关收敛
        self.rotation_lr = c.get("rotation_lr", None)
        self.save_dir = self.config["data"]["output"] + "/" + self.config["scene"]
        self.move_points = self.config["mapping"]["move_points"]
        self.online_plotting = self.config["mapping"]["online_plotting"]
        self.save_debug_visualization = self.config.get(
            "save_debug_visualization", False
        )
        # initialize_network 阶段：显式约束变形≈0，使 MLP 初始化为恒等变换，避免动态变化过大
        self.init_deform_reg_weight = c.get("init_deform_reg_weight", 10.0)
        # initialize_network 阶段将 deform 学习率乘以此系数，避免步长过大偏离 initial_map 结果（默认 0.1）
        self.init_network_deform_lr_scale = c.get("init_network_deform_lr_scale", 0.1)
        # initialize_network 前若干步仅优化零变形正则，再接入重建损失（默认 20）
        self.init_network_reg_warmup = c.get("init_network_reg_warmup", 20)
        # 为 True 时在 map() 中统计各阶段耗时并定期打印，用于分析瓶颈
        self.profile_mapping_time = c.get("profile_mapping_time", False)

    # --------------- 2. 初始化与关键帧 ---------------
    def add_next_kf(
        self, frame_idx, idx, viewpoint, init=False, scale=2.0, depth_map=None
    ):
        # This function computes the new Gaussians to be added given a new keyframe
        # print("depth",depth_map)
        self.gaussians.extend_from_pcd_seq(
            viewpoint,
            kf_id=frame_idx,
            init=init,
            scale=scale,
            depthmap=depth_map,
        )
        if frame_idx == self.dystart:
            self.gaussians.extend_from_pcd_seq(
                viewpoint,
                kf_id=frame_idx,
                init=True,
                scale=scale,
                depthmap=depth_map,
                add_dygs=True,
            )

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = True
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)

    # --------------- 3. 深度/位姿/尺度 ---------------
    def update_mapping_points(
        self, frame_idx, w2c, w2c_old, depth, depth_old, intrinsics, method=None
    ):
        """步骤1：rigid 则仅按 SE(3) 移动点；否则步骤2：投影取深度并缩放/位姿更新。"""
        if method == "rigid":
            frame_idxs = self.gaussians.unique_kfIDs
            frame_mask = frame_idxs == frame_idx
            if frame_mask.sum() == 0:
                return
            # Retrieve current set of points to be deformed
            # But first we need to retrieve all mean locations and clone them
            means = self.gaussians.get_xyz.detach()
            # Then move the points to their new location according to the new pose
            # The global transformation can be computed by composing the old pose
            # with the new pose
            transformation = torch.linalg.inv(torch.linalg.inv(w2c_old) @ w2c)
            pix_ones = torch.ones(frame_mask.sum(), 1).cuda().float()
            pts4 = torch.cat((means[frame_mask], pix_ones), dim=1)
            means[frame_mask] = (transformation @ pts4.T).T[:, :3]
            # put the new means back to the optimizer
            self.gaussians._xyz = self.gaussians.replace_tensor_to_optimizer(
                means, "xyz"
            )["xyz"]
            # transform the corresponding rotation matrices
            rots = self.gaussians.get_rotation.detach()
            # Convert transformation to quaternion
            transformation = rotation_matrix_to_quaternion(transformation.unsqueeze(0))
            rots[frame_mask] = quaternion_multiply(
                transformation.expand_as(rots[frame_mask]), rots[frame_mask]
            )

            with torch.no_grad():
                self.gaussians._rotation = self.gaussians.replace_tensor_to_optimizer(
                    rots, "rotation"
                )["rotation"]
        else:
            # Update pose and depth by projecting points into the pixel space to find updated correspondences.
            # This strategy also adjusts the scale of the gaussians to account for the distance change from the camera

            depth = depth.to(self.device)
            frame_idxs = (
                self.gaussians.unique_kfIDs
            )  # idx which anchored the set of points
            frame_mask = frame_idxs == frame_idx  # global variable
            if frame_mask.sum() == 0:
                return

            # Retrieve current set of points to be deformed
            means = self.gaussians.get_xyz.detach()[frame_mask]

            # Project the current means into the old camera to get the pixel locations
            pix_ones = torch.ones(means.shape[0], 1).cuda().float()
            pts4 = torch.cat((means, pix_ones), dim=1)
            pixel_locations = (intrinsics @ (w2c_old @ pts4.T)[:3, :]).T
            pixel_locations[:, 0] /= pixel_locations[:, 2]
            pixel_locations[:, 1] /= pixel_locations[:, 2]
            pixel_locations = pixel_locations[:, :2].long()
            height, width = depth.shape
            # Some pixels may project outside the viewing frustum.
            # Assign these pixels the depth of the closest border pixel
            pixel_locations[:, 0] = torch.clamp(
                pixel_locations[:, 0], min=0, max=width - 1
            )
            pixel_locations[:, 1] = torch.clamp(
                pixel_locations[:, 1], min=0, max=height - 1
            )

            # Extract the depth at those pixel locations from the new depth
            depth = depth[pixel_locations[:, 1], pixel_locations[:, 0]]
            pixel_locations = pixel_locations.to(depth_old.device)
            depth_old = depth_old[pixel_locations[:, 1], pixel_locations[:, 0]]
            # Next, we can either move the points to the new pose and then adjust the
            # depth or the other way around.
            # Lets adjust the depth per point first
            # First we need to transform the global means into the old camera frame
            pix_ones = torch.ones(frame_mask.sum(), 1).cuda().float()
            pts4 = torch.cat((means, pix_ones), dim=1)
            means_cam = (w2c_old @ pts4.T).T[:, :3]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            means_cam = means_cam.to(device)
            depth = depth.to(device)
            depth_old = depth_old.to(device)
            rescale_scale = (1 + 1 / (means_cam[:, 2]) * (depth - depth_old)).unsqueeze(
                -1
            )  # shift

            rigid_mask = torch.logical_or(depth == 0, depth_old == 0)
            rescale_scale[rigid_mask] = 1
            if (rescale_scale <= 0.0).sum() > 0:
                rescale_scale[rescale_scale <= 0.0] = 1

            rescale_mean = rescale_scale.repeat(1, 3)
            means_cam = rescale_mean * means_cam

            # Transform back means_cam to the world space
            pts4 = torch.cat((means_cam, pix_ones), dim=1)
            means = (torch.linalg.inv(w2c_old) @ pts4.T).T[:, :3]

            # Then move the points to their new location according to the new pose
            # The global transformation can be computed by composing the old pose
            # with the new pose
            transformation = torch.linalg.inv(torch.linalg.inv(w2c_old) @ w2c)
            pts4 = torch.cat((means, pix_ones), dim=1)
            means = (transformation @ pts4.T).T[:, :3]

            # reassign the new means of the frame mask to the self.gaussian object
            global_means = self.gaussians.get_xyz.detach()
            global_means[frame_mask] = means
            # print("mean nans: ", global_means.isnan().sum()/global_means.numel())
            self.gaussians._xyz = self.gaussians.replace_tensor_to_optimizer(
                global_means, "xyz"
            )["xyz"]

            # update the rotation of the gaussians
            rots = self.gaussians.get_rotation.detach()
            # Convert transformation to quaternion
            transformation = rotation_matrix_to_quaternion(transformation.unsqueeze(0))
            rots[frame_mask] = quaternion_multiply(
                transformation.expand_as(rots[frame_mask]), rots[frame_mask]
            )
            self.gaussians._rotation = self.gaussians.replace_tensor_to_optimizer(
                rots, "rotation"
            )["rotation"]

            # Update the scale of the Gaussians
            scales = self.gaussians._scaling.detach()
            scales[frame_mask] = scales[frame_mask] + torch.log(rescale_scale)
            self.gaussians._scaling = self.gaussians.replace_tensor_to_optimizer(
                scales, "scaling"
            )["scaling"]

    # --------------- 4. 几何与图像 ---------------
    def init_image_coor(self, height, width):
        x_row = np.arange(0, width)
        x = np.tile(x_row, (height, 1))
        x = x[np.newaxis, :, :]
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy()).cuda()
        u_u0 = x - width / 2.0

        y_col = np.arange(0, height)  # y_col = np.arange(0, height)
        y = np.tile(y_col, (width, 1)).T
        y = y[np.newaxis, :, :]
        y = y.astype(np.float32)
        y = torch.from_numpy(y.copy()).cuda()
        v_v0 = y - height / 2.0
        return u_u0, v_v0

    def depth_to_xyz(self, depth, focal_x, focal_y):
        b, c, h, w = depth.shape
        u_u0, v_v0 = self.init_image_coor(h, w)
        x = u_u0 * depth / focal_x
        y = v_v0 * depth / focal_y
        z = depth
        pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1)  # [b, h, w, c]
        # print(pw.shape)
        return pw

    def optimize_st(
        self,
        depth_torch,
        normal_torch,
        self_defined_focal_x: float,
        self_defined_focal_y: float,
    ):
        input_depth = depth_torch.cuda()
        input_depth = input_depth.unsqueeze(0).unsqueeze(0)
        gt_normal = normal_torch.cuda()
        focal_x = torch.Tensor(
            [
                self_defined_focal_x,
            ]
        ).cuda()
        focal_y = torch.Tensor(
            [
                self_defined_focal_y,
            ]
        ).cuda()
        s = nn.Parameter(torch.Tensor([1.0]).cuda().requires_grad_(True))
        t = nn.Parameter(torch.Tensor([0.0]).cuda().requires_grad_(True))

        optimizer = torch.optim.Adam(
            [
                {"params": s, "lr": 1e-3},
                {"params": t, "lr": 1e-3},
            ]
        )
        last_step_loss = 100000.0
        for step in range(500):
            optimizer.zero_grad()
            scaled_depth = s * input_depth + t
            # depth = s*input_depth+t
            depth_filter = nn.functional.avg_pool2d(
                scaled_depth, kernel_size=3, stride=1, padding=1
            )
            depth_filter = nn.functional.avg_pool2d(
                depth_filter, kernel_size=3, stride=1, padding=1
            )
            xyz = self.depth_to_xyz(depth_filter, focal_x, focal_y)
            xyz_i = xyz[0, :][None, :, :, :]
            pre_normal = self.get_surface_normalv2(xyz_i).permute((3, 2, 0, 1))
            similarity = torch.nn.functional.cosine_similarity(
                pre_normal, -gt_normal, dim=1
            )
            # if similarity
            loss = torch.nanmean(1 - similarity)

            loss.backward()
            optimizer.step()
            if step % 40 == 0:
                if abs(loss.item() - last_step_loss) < 1e-5:
                    break
                last_step_loss = loss.item()
        del scaled_depth, depth_filter, xyz, pre_normal
        torch.cuda.empty_cache()
        return s.item(), t.item()

    def get_surface_normalv2(self, xyz, patch_size=3):
        """
        xyz: xyz coordinates
        patch: [p1, p2, p3,
                p4, p5, p6,
                p7, p8, p9]
        surface_normal = [(p9-p1) x (p3-p7)] + [(p6-p4) - (p8-p2)]
        return: normal [h, w, 3, b]
        """
        eps = 1e-8
        b, h, w, c = xyz.shape
        half_patch = patch_size // 2
        xyz_pad = torch.zeros((b, h + patch_size - 1, w + patch_size - 1, c), dtype=xyz.dtype, device=xyz.device)
        xyz_pad[:, half_patch:-half_patch, half_patch:-half_patch, :] = xyz


        xyz_left = xyz_pad[:, half_patch:half_patch + h, :w, :]  # p4
        xyz_right = xyz_pad[:, half_patch:half_patch + h, -w:, :]  # p6
        xyz_top = xyz_pad[:, :h, half_patch:half_patch + w, :]  # p2
        xyz_bottom = xyz_pad[:, -h:, half_patch:half_patch + w, :]  # p8
        xyz_horizon = xyz_left - xyz_right  # p4p6
        xyz_vertical = xyz_top - xyz_bottom  # p2p8

        xyz_left_in = xyz_pad[:, half_patch:half_patch + h, 1:w + 1, :]  # p4
        xyz_right_in = xyz_pad[:, half_patch:half_patch + h, patch_size - 1:patch_size - 1 + w, :]  # p6
        xyz_top_in = xyz_pad[:, 1:h + 1, half_patch:half_patch + w, :]  # p2
        xyz_bottom_in = xyz_pad[:, patch_size - 1:patch_size - 1 + h, half_patch:half_patch + w, :]  # p8
        xyz_horizon_in = xyz_left_in - xyz_right_in  # p4p6
        xyz_vertical_in = xyz_top_in - xyz_bottom_in  # p2p8

        n_img_1 = torch.cross(xyz_horizon_in, xyz_vertical_in, dim=3)
        n_img_2 = torch.cross(xyz_horizon, xyz_vertical, dim=3)

        # re-orient normals consistently
        orient_mask = torch.sum(n_img_1 * xyz, dim=3) > 0
        n_img_1[orient_mask] *= -1
        orient_mask = torch.sum(n_img_2 * xyz, dim=3) > 0
        n_img_2[orient_mask] *= -1

        n_img1_L2 = torch.sqrt(torch.sum(n_img_1 ** 2, dim=3, keepdim=True)+ eps)
        n_img1_norm = n_img_1 / (n_img1_L2 + 1e-8)

        n_img2_L2 = torch.sqrt(torch.sum(n_img_2 ** 2, dim=3, keepdim=True)+ eps)
        n_img2_norm = n_img_2 / (n_img2_L2 + 1e-8)

        # average 2 norms
        n_img_aver = n_img1_norm + n_img2_norm
        n_img_aver_L2 = torch.sqrt(torch.sum(n_img_aver ** 2, dim=3, keepdim=True)+ eps)
        n_img_aver_norm = n_img_aver / (n_img_aver_L2 + 1e-8)
        # re-orient normals consistently
        orient_mask = torch.sum(n_img_aver_norm * xyz, dim=3) > 0
        n_img_aver_norm[orient_mask] *= -1
        n_img_aver_norm_out = n_img_aver_norm.permute((1, 2, 3, 0))  # [h, w, c, b]

        return n_img_aver_norm_out  # n_img1_norm.permute((1, 2, 3, 0))

    def obtain_allimgs_st(
        self,
        depth,
        normal,
        st_dict,
        video_idx,
        self_define_focal_x: float,
        self_define_focal_y: float,
    ):
        depth_np = depth.cpu().numpy()
        H, W = depth_np.shape
        depth_torch = torch.from_numpy(depth_np)  # (B, h, w)
        normal_torch = torch.from_numpy(normal)  # (B, h, w,3)
        if video_idx not in st_dict:
            s, t = self.optimize_st(
                depth_torch, normal_torch, self_define_focal_x, self_define_focal_y
            )
            st_dict[video_idx] = {"scale": s, "shift": t}
            # print(".==========.", s, t)

    def align_all_frames(
        self, depth, video_id, st_predicted, new_scale_alignFrame0: dict, static_msk
    ):
        # depth_dirs = glob(os.path.join(base_dir,'GeoWizardOut/depth_npy/*.npy'))
        if video_id == 11 and len(self.list) == 0:
            self.list.append(depth)
        reference_depth = self.list[0]

        reference_depth = reference_depth.cpu().numpy()
        # masked_reference_depth = reference_depth
        reference_s = st_predicted[11]["scale"]
        if st_predicted[11]["shift"] < 0:
            reference_t = -1.1 * st_predicted[11]["shift"]  # 0.9
        else:
            reference_t = st_predicted[11]["shift"]

        scaled_refer_depth_nomsk = reference_s * reference_depth + reference_t
        h, w = scaled_refer_depth_nomsk.shape
        static_msk = static_msk.cpu().numpy()
        static_msk = np.array(Image.fromarray(static_msk).resize((w, h))) > 0

        scaled_refer_depth = scaled_refer_depth_nomsk[static_msk]
        del reference_t, reference_s

        Y = torch.from_numpy(scaled_refer_depth).unsqueeze(-1)

        depth = depth
        cur_s = st_predicted[video_id]["scale"]

        if self.longer:
            cur_t = max(st_predicted[video_id]["shift"], 0)
        else:
            if st_predicted[video_id]["shift"] < 0:
                cur_t = -st_predicted[video_id]["shift"]
            else:
                cur_t = st_predicted[video_id]["shift"]
        cur_masked_depth = depth[static_msk]
        if not np.isnan(cur_s) or not np.isnan(cur_t):
            pass
        else:
            cur_s = st_predicted["mean_s"]
            cur_t = st_predicted["mean_t"]
        scaled_cur_depth = cur_masked_depth * cur_s + cur_t
        # print()
        scaled_cur_depth = scaled_cur_depth.cpu().numpy()
        ##### Solving using
        A = torch.from_numpy(scaled_cur_depth).unsqueeze(-1)

        res = torch.linalg.lstsq(A, Y)
        if res.solution.item() == 0:
            new_scale_alignFrame0[video_id] = 1.0
        else:
            new_scale_alignFrame0[video_id] = res.solution.item()

            previous_value = new_scale_alignFrame0[11]

            if previous_value is not None and self.longer:
                current_value = new_scale_alignFrame0[video_id]

                if (
                    current_value is None
                    or current_value > previous_value * self.s
                    or current_value * self.s < previous_value
                ):
                    new_scale_alignFrame0[video_id] = previous_value
                    # print(f"new，video_id={video_id} is: {new_scale_alignFrame0[video_id]}")
                else:
                    # print(f"old，video_id={video_id} is: {current_value}")
                    pass
            else:
                print("no last key")
        # print("S,T,align_scale:", cur_s, cur_t, new_scale_alignFrame0[video_id])
        del cur_s, cur_t, res
        return new_scale_alignFrame0
        pass

    def export_scaled_pcd(
        self,
        st_predicted,
        new_scale_alignFrame0,
        depth,
        video_id,
        mean_st=False,
    ):

        # import open3d as o3d

        if mean_st:
            s = st_predicted["mean_s"]
            t = st_predicted["mean_t"]
        else:
            s = st_predicted[video_id]["scale"]

            if st_predicted[video_id]["shift"] < 0:
                t = -self.shift * st_predicted[video_id]["shift"]  # 0.6
            else:
                t = st_predicted[video_id]["shift"]
            # t = max(st_predicted[video_id]["shift"], 0)
        depth = depth * s + t
        if video_id in new_scale_alignFrame0:
            depth = depth * new_scale_alignFrame0[video_id]
        return depth

    def get_depth_order_loss(
        self,
        render_depth,
        gt_depth,
        mask,
        pair_num=200000,
        alpha=100,
    ):
        """_summary_

        Args:
            render_depth (_type_): 1,H,W,
            gt_depth (_type_): H,W,
            mask (_type_): H,W,
            method_name (str, optional): _description_. Defaults to "pearson".

        Returns:
            _type_: _description_
        """
        if isinstance(gt_depth, np.ndarray):
            gt_depth = torch.from_numpy(gt_depth).to(render_depth.device)
        gt_depth = gt_depth.to(render_depth.device)
        # alpha = 100
        gt_depth = gt_depth[mask > 0]  ## N,1
        depthmax = gt_depth.max()
        depthmin = gt_depth.min()
        interval = (depthmax - depthmin) / 10
        # interval = (depthmax-depthmin)/20

        render_depth = render_depth.squeeze(0)[mask > 0]  ## N,1
        index1 = torch.randperm(gt_depth.shape[0])[:pair_num,]
        index2 = torch.randperm(gt_depth.shape[0])[:pair_num,]
        index1 = index1.to(render_depth.device)
        index2 = index2.to(render_depth.device)
        threshold_msk = torch.abs(gt_depth[index1] - gt_depth[index2]) >= interval
        threshold_msk = threshold_msk.to(render_depth.device)
        index1 = index1[threshold_msk]
        index2 = index2[threshold_msk]

        gt_oder = torch.sign(gt_depth[index1] - gt_depth[index2])
        render_diff = render_depth[index1] - render_depth[index2]

        loss = torch.mean(torch.abs(torch.tanh(alpha * render_diff) - gt_oder))
        return loss

    def get_w2c_and_depth(
        self,
        video_idx,
        idx,
        mono_depth,
        motion_mask,
        depth_gt,
        normal,
        mono,
        static_mask,
        print_info=False,
        init=False,
    ):
        """步骤：从 video 取位姿与深度→无效则返回 invalid；否则单目深度腐蚀/修复→s,t 对齐→export_scaled_pcd 返回深度与 w2c。"""

        est_droid_depth, valid_depth_mask, c2w = self.video.get_depth_and_pose(
            video_idx, self.device
        )

        c2w = c2w.to(self.device)
        w2c = torch.linalg.inv(c2w)

        if print_info:
            print(
                f"valid depth number: {valid_depth_mask.sum().item()}, "
                f"valid depth ratio: {(valid_depth_mask.sum() / (valid_depth_mask.shape[0] * valid_depth_mask.shape[1])).item()}"
            )

        if valid_depth_mask.sum() < 100:
            invalid = True
            print(
                f"Skip mapping frame {idx} at video idx {video_idx} because of not enough valid depth ({valid_depth_mask.sum()})."
            )
        else:
            invalid = False
            est_droid_depth[~valid_depth_mask] = 0

            mono_valid_mask = mono_depth < (mono_depth.mean() * 3)
            mono_depth[mono_depth > 3 * mono_depth.mean()] = 0

            from scipy.ndimage import binary_erosion

            mono_depth = mono_depth.cpu().numpy()
            binary_image = (mono_depth > 0).astype(int)
            iterations = 5
            padded_binary_image = np.pad(
                binary_image, pad_width=iterations, mode="constant", constant_values=1
            )
            structure = np.ones((3, 3), dtype=int)

            eroded_padded_image = binary_erosion(
                padded_binary_image, structure=structure, iterations=iterations
            )
            eroded_image = eroded_padded_image[
                iterations:-iterations, iterations:-iterations
            ]

            mono_depth[eroded_image == 0] = 0

            if (mono_depth == 0).sum() > 0:
                mono_depth = torch.from_numpy(
                    cv2.inpaint(
                        mono_depth,
                        (mono_depth == 0).astype(np.uint8),
                        inpaintRadius=3,
                        flags=cv2.INPAINT_NS,
                    )
                ).to(self.device)
            else:
                mono_depth = torch.from_numpy(mono_depth).to(self.device)
            normal = normal.cpu().numpy()
            depth_gt = torch.from_numpy(depth_gt).to(self.device)

            if video_idx == 11 and len(self.first_d) == 0:
                self.first_d.append(motion_mask)
            refer_mask = self.first_d[0]
            self.obtain_allimgs_st(
                mono_depth, normal, self.st_predicted, video_idx, 535.4, 539.2
            )
            mean_s = 0
            mean_t = 0
            invalid = 0

            self.st_predicted["mean_s"] = 0.8
            self.st_predicted["mean_t"] = 0.15
            self.static_msk = motion_mask * refer_mask
            self.align_all_frames(
                mono_depth,
                video_idx,
                self.st_predicted,
                self.new_scale_alignFrame0,
                self.static_msk,
            )
            mono_depth = self.export_scaled_pcd(
                self.st_predicted,
                self.new_scale_alignFrame0,
                mono_depth,
                video_idx,
                mean_st=False,
            )
            mono_depth_wq = mono_depth
            torch.cuda.empty_cache()
        return mono_depth_wq, w2c, invalid

    def initialize_map(self, cur_frame_idx, idx, viewpoint):
        """步骤1：多轮渲染→解包→loss→反传；步骤2：无梯度下 max_radii2D/densify/reset_opacity/optimizer.step；步骤3：occ_aware_visibility 与调试图；步骤4：可选 online 绘图。"""
        from src.utils.Printer import get_msg_prefix

        prefix = get_msg_prefix(FontColor.MAPPER)
        pbar = tqdm(
            range(self.init_itr_num),
            desc=prefix + "Initializing map",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            position=0,
            leave=True,
        )

        # 仅在使用法向/扁平损失时覆盖 rotation 学习率，与 copy 行为一致
        flatten_w = self.flattening_loss_weight
        if flatten_w > 0 and self.rotation_lr is not None:
            for pg in self.gaussians.optimizer.param_groups:
                if pg.get("name") == "rotation":
                    pg["lr"] = float(self.rotation_lr)
                    break

        for mapping_iteration in pbar:
            self.iteration_count += 1
            # 步骤1：静态渲染（return_normal=False 与 copy 一致，避免多余法向计算影响收敛）
            render_pkg = self._render_viewpoint(
                viewpoint, 0, 0, None, None, None, return_normal=False
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
                normal,
            ) = self._unpack_render_pkg(render_pkg)
            loss_init = get_loss_mapping(
                self.config["mapping"],
                image,
                depth,
                viewpoint,
                opacity,
                initialization=True,
                rm_dynamic=not (self.dystart == cur_frame_idx),
            )
            # if flatten_w > 0 and normal is not None:
            #     loss_init += flatten_w * self.get_loss_flattening(
            #         normal, viewpoint, opacity=opacity
            #     )
            # if flatten_w > 0:
            #     loss_init += self.scale_loss_weight * self.get_loss_flat(
            #         visibility_filter
            #     )
            loss_init.backward()

            # 步骤2：无梯度下更新 max_radii2D、densification_stats、densify_and_prune、reset_opacity、optimizer.step
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )
                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
        # 步骤3：写 occ_aware_visibility、保存调试图、打印 Initialized map
        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        vis_render_process(
            self.gaussians,
            self.pipeline_params,
            self.background,
            viewpoint,
            viewpoint.uid,
            self.save_dir,
            out_dir="map",
            mask=None,
            dynamic=False,
            save_debug=getattr(self, "save_debug_visualization", False),
        )

        # online plotting
        if self.online_plotting:
            from thirdparty.gaussian_splatting.utils.image_utils import psnr
            from src.utils.eval_utils import plot_rgbd_silhouette
            import cv2
            import numpy as np

            cur_idx = self.current_window[np.array(self.current_window).argmax()]
            viewpoint = self.viewpoints[cur_idx]
            render_pkg = self._render_viewpoint(viewpoint, 0, 0, None, None, None)
            image = render_pkg["render"].detach()
            depth = render_pkg["depth"].detach()
            gt_image = viewpoint.original_image
            gt_depth = viewpoint.depth
            image = torch.clamp(image, 0.0, 1.0)
            gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
            pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
                np.uint8
            )
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
            mask = gt_image > 0
            psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
            diff_depth_l1 = torch.abs(depth.detach().cpu() - gt_depth)
            diff_depth_l1 = diff_depth_l1 * (gt_depth > 0)
            depth_l1 = diff_depth_l1.sum() / (gt_depth > 0).sum()
            # Add plotting 2x3 grid here
            plot_dir = self.save_dir + "/online_plots"
            plot_rgbd_silhouette(
                gt_image,
                gt_depth,
                image,
                depth,
                diff_depth_l1,
                psnr_score.item(),
                depth_l1,
                plot_dir=plot_dir,
                idx=str(cur_idx),
                diff_rgb=np.abs(gt - pred),
            )
        return render_pkg

    # --------------- 5. 变形与渲染 ---------------
    def find_closest_keyframe(self, uid):
        prev_keys = [key for key in self.viewpoints if key < uid]
        if prev_keys:
            return max(prev_keys)

        next_keys = [key for key in self.viewpoints if key > uid]
        if next_keys:
            return min(next_keys)

        return None

    # --------------- 映射阶段复用：变形量 / 渲染 / 光流损失 / 正则 ---------------
    def _get_deform_d_values(
        self, viewpoint, use_noise=False, detach_outputs=False, iteration=0
    ):
        """步骤：按 viewpoint.fid 取 time_input，可选加噪声，调用 deform.step，返回 d_xyz/d_rotation/d_scaling/d_opacity/d_color。"""
        time_input = self.gaussians.deform.deform.expand_time(viewpoint.fid)
        if use_noise:
            N = time_input.shape[0]
            ast_noise = (
                torch.randn(1, 1, device=time_input.device).expand(N, -1)
                * self.gaussians.time_interval
                * self.gaussians.smooth_term(self.iteration_count)
            )
            d_values = self.gaussians.deform.step(
                self.gaussians.get_xyz.detach(),
                time_input + ast_noise,
                iteration=iteration,
                feature=None,
                motion_mask=self.gaussians.motion_mask,
                camera_center=viewpoint.camera_center,
                time_interval=self.gaussians.time_interval,
            )
        else:
            d_values = self.gaussians.deform.step(
                self.gaussians.get_dygs_xyz.detach(),
                time_input,
                iteration=iteration,
                feature=None,
                motion_mask=self.gaussians.motion_mask,
                camera_center=viewpoint.camera_center,
                time_interval=self.gaussians.time_interval,
            )
        if detach_outputs:
            dxyz = d_values["d_xyz"].detach()
            d_rot = d_values["d_rotation"].detach()
            d_scale = d_values["d_scaling"].detach()
            d_opac = (
                d_values["d_opacity"].detach()
                if d_values.get("d_opacity") is not None
                else None
            )
            d_color = (
                d_values["d_color"].detach()
                if d_values.get("d_color") is not None
                else None
            )
        else:
            dxyz = d_values["d_xyz"]
            d_rot = d_values["d_rotation"]
            d_scale = d_values["d_scaling"]
            d_opac = d_values.get("d_opacity")
            d_color = d_values.get("d_color")
        return dxyz, d_rot, d_scale, d_opac, d_color

    def _get_deform_d_values_for_fid(self, fid, camera_center, iteration=0):
        """步骤：按给定 fid、camera_center 调用 deform.step，返回完整 d_value 字典（用于最近关键帧）。"""
        time_input = self.gaussians.deform.deform.expand_time(fid)
        d_value2 = self.gaussians.deform.step(
            self.gaussians.get_dygs_xyz.detach(),
            time_input,
            iteration=iteration,
            feature=None,
            motion_mask=self.gaussians.motion_mask,
            camera_center=camera_center,
            time_interval=self.gaussians.time_interval,
        )
        return d_value2

    def _render_viewpoint(
        self,
        viewpoint,
        dxyz,
        d_scale,
        d_rot,
        d_opac=None,
        d_color=None,
        return_normal=False,
    ):
        """步骤：用 dx/ds/dr/do/dc 调用 render，返回 render_pkg。return_normal=True 时额外渲染最短轴法向量供扁平化损失与多视角单应使用。"""
        return render(
            viewpoint,
            self.gaussians,
            self.pipeline_params,
            self.background,
            dynamic=False,
            dx=dxyz,
            ds=d_scale,
            dr=d_rot,
            do=d_opac,
            dc=d_color,
            return_normal=return_normal,
        )

    def _unpack_render_pkg(self, render_pkg):
        """步骤：解包 render/render_pkg 为 image, viewspace_points, visibility_filter, radii, depth, opacity, n_touched, normal。"""
        return (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
            render_pkg["depth"],
            render_pkg["opacity"],
            render_pkg["n_touched"],
            render_pkg.get("normal", None),
        )

    def _compute_flow_loss_pair(
        self, viewpoint, closest_viewpoint, idx1, dxyz, d_rot, d_scale, flow_weights
    ):
        """步骤：generate_flow→取最近帧 d_value2→render_flow 双向→按 motion_mask 加权 L1，返回总光流损失与 d_value2。"""
        flow, flow_back, mask_fwd, mask_bwd, _ = viewpoint.generate_flow(
            viewpoint.original_image.cuda(),
            idx1,
            closest_viewpoint.original_image.cuda(),
        )
        d_value2 = self._get_deform_d_values_for_fid(
            closest_viewpoint.fid,
            closest_viewpoint.camera_center,
            iteration=0,
        )
        d_xyz2 = d_value2["d_xyz"]
        render_pkg2 = render_flow(
            pc=self.gaussians,
            viewpoint_camera1=viewpoint,
            viewpoint_camera2=closest_viewpoint,
            d_xyz1=dxyz,
            d_xyz2=d_xyz2,
            d_rotation1=d_rot,
            d_scaling1=d_scale,
            scale_const=None,
        )
        coor1to2_motion = render_pkg2["render"][:2].permute(1, 2, 0)
        dynamic_mask = (
            (~viewpoint.motion_mask)
            .unsqueeze(0)
            .permute(1, 2, 0)
            .repeat(1, 1, 2)
            .detach()
        )
        loss_fwd = flow_weights * l1_loss(
            flow_back * dynamic_mask, coor1to2_motion * dynamic_mask
        )
        render_pkg_back = render_flow(
            pc=self.gaussians,
            viewpoint_camera1=closest_viewpoint,
            viewpoint_camera2=viewpoint,
            d_xyz1=d_xyz2,
            d_xyz2=dxyz,
            d_rotation1=d_value2["d_rotation"],
            d_scaling1=d_value2["d_scaling"],
        )
        coor2to1_motion = render_pkg_back["render"][:2].permute(1, 2, 0)
        dynamic_mask_b = (
            (~closest_viewpoint.motion_mask)
            .unsqueeze(0)
            .permute(1, 2, 0)
            .repeat(1, 1, 2)
            .detach()
        )
        loss_bwd = flow_weights * l1_loss(
            flow * dynamic_mask_b, coor2to1_motion * dynamic_mask_b
        )
        return loss_fwd + loss_bwd, d_value2

    # --------------- 6. 损失 ---------------
    def _add_deform_regularization_losses(
        self, viewpoint, delta=None, scale_arap=1e-3, scale_acc=1e-5, scale_elastic=1e-3
    ):
        """步骤：按 viewpoint.fid 计算 arap_loss、acc_loss、elastic_loss 并加权求和。"""
        delta_t = (delta or 5) * self.gaussians.time_interval
        loss = scale_arap * self.gaussians.deform.deform.arap_loss(
            t=viewpoint.fid,
            delta_t=delta_t,
            t_samp_num=4,
        )
        loss = loss + scale_acc * self.gaussians.deform.deform.acc_loss(
            t=viewpoint.fid,
            delta_t=5 * self.gaussians.time_interval,
        )
        loss = loss + scale_elastic * self.gaussians.deform.deform.elastic_loss(
            t=viewpoint.fid,
            delta_t=5 * self.gaussians.time_interval,
        )
        return loss

    # --------------- 7. 多视角/NCC（map 依赖） ---------------
    def lncc(self, ref, nea):
        # ref_gray: [batch_size, total_patch_size]
        # nea_grays: [batch_size, total_patch_size]
        bs, tps = nea.shape
        patch_size = int(np.sqrt(tps))

        ref_nea = ref * nea
        ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
        ref = ref.view(bs, 1, patch_size, patch_size)
        nea = nea.view(bs, 1, patch_size, patch_size)
        ref2 = ref.pow(2)
        nea2 = nea.pow(2)

        # sum over kernel
        filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
        padding = patch_size // 2
        ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[
            :, :, padding, padding
        ]
        nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[
            :, :, padding, padding
        ]
        ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[
            :, :, padding, padding
        ]
        nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[
            :, :, padding, padding
        ]
        ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[
            :, :, padding, padding
        ]

        # average over kernel
        ref_avg = ref_sum / tps
        nea_avg = nea_sum / tps

        cross = ref_nea_sum - nea_avg * ref_sum
        ref_var = ref2_sum - ref_avg * ref_sum
        nea_var = nea2_sum - nea_avg * nea_sum

        cc = cross * cross / (ref_var * nea_var + 1e-8)
        ncc = 1 - cc
        ncc = torch.clamp(ncc, 0.0, 2.0)
        ncc = torch.mean(ncc, dim=1, keepdim=True)
        mask = ncc < 0.9
        return ncc, mask

    def patch_warp(self, H, uv):
        B, P = uv.shape[:2]
        H = H.view(B, 3, 3)
        ones = torch.ones((B, P, 1), device=uv.device)
        homo_uv = torch.cat((uv, ones), dim=-1)

        grid_tmp = torch.einsum("bik,bpk->bpi", H, homo_uv)
        grid_tmp = grid_tmp.reshape(B, P, 3)
        grid = grid_tmp[..., :2] / (grid_tmp[..., 2:] + 1e-10)
        return grid

    def patch_offsets(self, h_patch_size, device):
        offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
        return torch.stack(
            torch.meshgrid(offsets, offsets, indexing="xy")[::-1], dim=-1
        ).view(1, -1, 2)

    def _compute_normal_from_rendered_depth(self, depth, viewpoint_cam):
        """从渲染深度估计法向（可微），用于无渲染法向时的多视角单应。"""
        if depth.dim() == 3:
            depth_map = depth.squeeze(0)
        else:
            depth_map = depth

        device = depth_map.device
        dtype = depth_map.dtype
        H, W = depth_map.shape

        fx = torch.tensor(viewpoint_cam.fx, device=device, dtype=dtype)
        fy = torch.tensor(viewpoint_cam.fy, device=device, dtype=dtype)
        cx = torch.tensor(viewpoint_cam.cx, device=device, dtype=dtype)
        cy = torch.tensor(viewpoint_cam.cy, device=device, dtype=dtype)

        u, v = torch.meshgrid(
            torch.arange(W, device=device, dtype=dtype),
            torch.arange(H, device=device, dtype=dtype),
            indexing="xy",
        )
        x = (u - cx) / fx * depth_map
        y = (v - cy) / fy * depth_map
        pts = torch.stack([x, y, depth_map], dim=-1)

        pts_l = torch.roll(pts, shifts=1, dims=1)
        pts_r = torch.roll(pts, shifts=-1, dims=1)
        pts_u = torch.roll(pts, shifts=1, dims=0)
        pts_d = torch.roll(pts, shifts=-1, dims=0)

        normal_hwc = torch.cross(pts_r - pts_l, pts_d - pts_u, dim=-1)
        normal_hwc = F.normalize(normal_hwc, dim=-1, eps=1e-6)

        valid = depth_map > 1e-6
        valid = (
            valid
            & torch.roll(valid, 1, 0)
            & torch.roll(valid, -1, 0)
            & torch.roll(valid, 1, 1)
            & torch.roll(valid, -1, 1)
        )
        valid[0, :] = False
        valid[-1, :] = False
        valid[:, 0] = False
        valid[:, -1] = False

        normal_hwc = torch.where(
            valid.unsqueeze(-1), normal_hwc, torch.zeros_like(normal_hwc)
        )
        return normal_hwc.permute(2, 0, 1)

    def compute_multi_view_loss(
        self,
        viewpoint_cam,
        render_pkg,
        gaussians,
        pipe,
        bg,
        dxyz,
        d_scale,
        d_rot,
        d_opac,
        d_color,
        nearest_cam,
    ):
        """步骤：渲染最近视角→当前深度投影到最近视角→深度校正→重投影回当前像平面→几何掩码 d_mask→与 motion_mask 结合→NCC 损失。"""
        if nearest_cam is None:
            return 0.0
        use_virtul_cam = False
        # 获取配置参数
        patch_size = 3
        sample_num = 102400
        pixel_noise_th = 1.0
        ncc_weight = 0.15
        # geo_weight = self.opt.multi_view_geo_weight
        total_patch_size = (patch_size * 2 + 1) ** 2
        gt_image, gt_image_gray = viewpoint_cam.get_image()
        # 初始化损失
        total_loss = 0.0

        try:
            ## 计算几何一致性掩码和损失
            # 检查是否有动态掩码，如果两个视角都没有动态掩码，则跳过多视角损失计算
            has_current_motion_mask = viewpoint_cam.motion_mask is not None
            has_nearest_motion_mask = nearest_cam.motion_mask is not None

            H, W = render_pkg["depth"].squeeze().shape
            ix, iy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
            pixels = (
                torch.stack([ix, iy], dim=-1).float().to(render_pkg["depth"].device)
            )

            # 渲染最近视角
            nearest_render_pkg = render(
                nearest_cam,
                gaussians,
                pipe,
                bg,
                dynamic=False,
                dx=dxyz,
                ds=d_scale,
                dr=d_rot,
                do=d_opac,
                dc=d_color,
            )

            # 计算3D点投影
            pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg["depth"])
            pts_in_nearest_cam = pts @ nearest_cam.R_gt + nearest_cam.T_gt
            map_z, d_mask = gaussians.get_points_depth_in_depth_map(
                nearest_cam, nearest_render_pkg["depth"], pts_in_nearest_cam
            )

            # 深度校正
            pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:, 2:3])
            pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[..., None]

            # 坐标变换：最近视角相机坐标系 → 世界坐标系 → 当前视角相机坐标系
            R_wvt = nearest_cam.R_gt
            T_wvt = nearest_cam.T_gt
            pts_ = (pts_in_nearest_cam - T_wvt) @ R_wvt.transpose(-1, -2)
            pts_in_view_cam = pts_ @ viewpoint_cam.R_gt + viewpoint_cam.T_gt

            # 投影到图像平面
            pts_projections = torch.stack(
                [
                    pts_in_view_cam[:, 0] * viewpoint_cam.fx / pts_in_view_cam[:, 2]
                    + viewpoint_cam.cx,
                    pts_in_view_cam[:, 1] * viewpoint_cam.fy / pts_in_view_cam[:, 2]
                    + viewpoint_cam.cy,
                ],
                -1,
            ).float()

            # 计算像素噪声和权重
            pixel_noise = torch.norm(
                pts_projections - pixels.reshape(*pts_projections.shape), dim=-1
            )
            d_mask_before_motion = d_mask & (
                pixel_noise < pixel_noise_th
            )  # 保存应用motion_mask之前的d_mask
            d_mask = d_mask_before_motion.clone()

            # 将d_mask与motion_mask相乘（motion_mask=True表示静态区域）
            # 直接在一维空间操作，避免reshape开销

            # 与当前视角的静态掩码相乘
            current_motion_mask = None
            if has_current_motion_mask:
                current_motion_mask = viewpoint_cam.motion_mask.reshape(-1).to(
                    d_mask.device
                )
                d_mask = d_mask & current_motion_mask

            # 与临近视角的静态掩码相乘
            nearest_motion_mask = None
            if has_nearest_motion_mask:
                nearest_motion_mask = nearest_cam.motion_mask.reshape(-1).to(
                    d_mask.device
                )
                d_mask = d_mask & nearest_motion_mask

            # 可视化 d_mask 和 motion_mask 的结合效果
            if (
                hasattr(self, "visualize_mask_combination")
                and self.visualize_mask_combination
            ):
                self._visualize_mask_combination(
                    d_mask_before_motion,
                    current_motion_mask,
                    nearest_motion_mask,
                    d_mask,
                    H,
                    W,
                    viewpoint_cam,
                    render_pkg,
                    render_pkg["render"],
                )

            weights = (1.0 / torch.exp(pixel_noise)).detach()
            weights[~d_mask] = 0

            # 可视化 d_mask（可选，设置 visualize_mask=True 启用）
            if hasattr(self, "visualize_mask") and self.visualize_mask:
                self._visualize_d_mask(
                    d_mask, pixel_noise, H, W, viewpoint_cam, render_pkg
                )

            # 几何项：必须始终参与计算图，否则 multi_view 损失无法对 d_value2 / 当前 depth 反传。
            # 无有效 d_mask 时用全局 pixel_noise 均值作为 fallback，保证梯度路径存在。
            geo_fallback = 0.001 * pixel_noise.mean()
            total_loss = total_loss + geo_fallback

            # 有有效 d_mask 时再加 NCC（内部含 geo_loss + ncc_loss），与 fallback 一起保证反传
            if torch.any(d_mask) and use_virtul_cam is False:
                ncc_loss = self._compute_ncc_loss(
                    viewpoint_cam,
                    nearest_cam,
                    render_pkg,
                    gt_image_gray,
                    pixel_noise,
                    d_mask,
                    weights,
                    pixels,
                    patch_size,
                    sample_num,
                    total_patch_size,
                    ncc_weight,
                )
                total_loss = total_loss + ncc_loss
            return total_loss
            # return 100.0 * total_loss

        except Exception as e:
            print(f"Error in compute_multi_view_loss: {e}")
            return render_pkg["depth"].mean() * 0.0

    def _visualize_d_mask(self, d_mask, pixel_noise, H, W, viewpoint_cam, render_pkg):
        """
        可视化深度一致性掩码 d_mask

        参数:
            d_mask: 布尔掩码张量 [H*W]
            pixel_noise: 像素重投影误差 [H*W]
            H, W: 图像高度和宽度
            viewpoint_cam: 当前视角相机
            render_pkg: 渲染结果包
        """
        import matplotlib.pyplot as plt
        import os

        # 创建保存目录
        vis_dir = os.path.join(self.save_dir, "d_mask_visualization")
        os.makedirs(vis_dir, exist_ok=True)

        # 将掩码和噪声转换为图像格式 - 先全部转换为numpy
        d_mask_np = d_mask.reshape(H, W).detach().cpu().numpy()
        d_mask_img = (d_mask_np * 255).astype(np.uint8)

        pixel_noise_np = pixel_noise.reshape(H, W).detach().cpu().numpy()
        pixel_noise_flat = pixel_noise_np.reshape(-1)  # 展平为一维数组

        # 获取渲染的RGB图像和深度图
        rendered_image = render_pkg["render"].detach().cpu().permute(1, 2, 0).numpy()
        rendered_depth = render_pkg["depth"].squeeze().detach().cpu().numpy()

        # 创建可视化图像
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. 渲染的RGB图像
        axes[0, 0].imshow(rendered_image)
        axes[0, 0].set_title("渲染图像")
        axes[0, 0].axis("off")

        # 2. 渲染的深度图
        depth_vis = axes[0, 1].imshow(rendered_depth, cmap="turbo")
        axes[0, 1].set_title("渲染深度图")
        axes[0, 1].axis("off")
        plt.colorbar(depth_vis, ax=axes[0, 1])

        # 3. d_mask 掩码（黑白）
        axes[0, 2].imshow(d_mask_img, cmap="gray")
        axes[0, 2].set_title(f"d_mask (有效点: {d_mask.sum().item()}/{d_mask.numel()})")
        axes[0, 2].axis("off")

        # 4. d_mask 叠加在RGB图像上（绿色表示有效区域）
        overlay = rendered_image.copy()
        mask_colored = np.zeros_like(overlay)
        mask_colored[:, :, 1] = d_mask_img / 255.0  # 绿色通道
        axes[1, 0].imshow(overlay * 0.5 + mask_colored * 0.5)
        axes[1, 0].set_title("d_mask 叠加（绿色=有效）")
        axes[1, 0].axis("off")

        # 5. 像素重投影误差热力图
        noise_vis = axes[1, 1].imshow(pixel_noise_np, cmap="hot", vmin=0, vmax=5)
        axes[1, 1].set_title("像素重投影误差")
        axes[1, 1].axis("off")
        plt.colorbar(noise_vis, ax=axes[1, 1])

        # 6. 误差分布直方图 - 修复这里！
        d_mask_flat = d_mask_np.reshape(-1).astype(bool)  # 展平并转换为布尔数组
        valid_noise = pixel_noise_flat[d_mask_flat]
        invalid_noise = pixel_noise_flat[~d_mask_flat]

        axes[1, 2].hist(valid_noise, bins=50, alpha=0.5, label="有效点", color="green")
        axes[1, 2].hist(invalid_noise, bins=50, alpha=0.5, label="无效点", color="red")
        axes[1, 2].set_xlabel("重投影误差 (像素)")
        axes[1, 2].set_ylabel("频数")
        axes[1, 2].set_title("误差分布")
        axes[1, 2].legend()
        axes[1, 2].set_xlim(0, 10)

        plt.tight_layout()

        # 保存图像
        frame_id = viewpoint_cam.uid if hasattr(viewpoint_cam, "uid") else "unknown"
        save_path = os.path.join(vis_dir, f"d_mask_frame_{frame_id}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"[可视化] d_mask 已保存到: {save_path}")
        print(
            f"[统计] 有效点数: {d_mask.sum().item()} / {d_mask.numel()} ({100 * d_mask.sum().item() / d_mask.numel():.2f}%)"
        )
        print(f"[统计] 平均重投影误差 (有效): {valid_noise.mean():.3f} 像素")
        if len(invalid_noise) > 0:
            print(f"[统计] 平均重投影误差 (无效): {invalid_noise.mean():.3f} 像素")

    def _visualize_mask_combination(
        self,
        d_mask_before,
        current_motion_mask,
        nearest_motion_mask,
        d_mask_after,
        H,
        W,
        viewpoint_cam,
        render_pkg,
        gt_image,
    ):
        """
        可视化 d_mask 与 motion_mask 的结合效果

        参数:
            d_mask_before: 应用 motion_mask 之前的 d_mask [H*W]
            current_motion_mask: 当前视角的 motion_mask [H*W] 或 None
            nearest_motion_mask: 临近视角的 motion_mask [H*W] 或 None
            d_mask_after: 应用 motion_mask 之后的 d_mask [H*W]
            H, W: 图像高度和宽度
            viewpoint_cam: 当前视角相机
            render_pkg: 渲染结果包
            gt_image: 真实图像 [3, H, W]
        """
        import matplotlib.pyplot as plt
        import os

        # 创建保存目录
        vis_dir = os.path.join(self.save_dir, "mask_combination_visualization")
        os.makedirs(vis_dir, exist_ok=True)

        # 转换为 numpy 数组
        d_mask_before_np = d_mask_before.reshape(H, W).detach().cpu().numpy()
        d_mask_after_np = d_mask_after.reshape(H, W).detach().cpu().numpy()

        # 获取真实图像
        gt_image_np = gt_image.detach().cpu().permute(1, 2, 0).numpy()
        gt_image_np = np.clip(gt_image_np, 0, 1)

        # 获取渲染图像
        rendered_image = render_pkg["render"].detach().cpu().permute(1, 2, 0).numpy()
        rendered_image = np.clip(rendered_image, 0, 1)

        # 计算需要的行数
        num_rows = (
            3
            if (current_motion_mask is not None or nearest_motion_mask is not None)
            else 2
        )
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

        # 第一行：原始图像和 d_mask
        axes[0, 0].imshow(gt_image_np)
        axes[0, 0].set_title("真实图像")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(d_mask_before_np, cmap="gray")
        valid_before = d_mask_before.sum().item()
        axes[0, 1].set_title(
            f"d_mask (应用motion_mask前)\n有效点: {valid_before}/{d_mask_before.numel()} ({100 * valid_before / d_mask_before.numel():.1f}%)"
        )
        axes[0, 1].axis("off")

        # d_mask 叠加在真实图像上
        overlay_before = gt_image_np.copy()
        mask_overlay = np.zeros_like(overlay_before)
        mask_overlay[:, :, 1] = d_mask_before_np  # 绿色通道
        axes[0, 2].imshow(overlay_before * 0.6 + mask_overlay * 0.4)
        axes[0, 2].set_title("d_mask 叠加（绿色=有效）")
        axes[0, 2].axis("off")

        # 第二行：motion_mask
        row_idx = 1
        if current_motion_mask is not None:
            current_motion_np = current_motion_mask.reshape(H, W).detach().cpu().numpy()
            axes[row_idx, 0].imshow(
                current_motion_np, cmap="RdYlGn"
            )  # 红色=动态，绿色=静态
            static_current = current_motion_mask.sum().item()
            axes[row_idx, 0].set_title(
                f"当前视角 motion_mask\n静态点: {static_current}/{current_motion_mask.numel()} ({100 * static_current / current_motion_mask.numel():.1f}%)"
            )
            axes[row_idx, 0].axis("off")
        else:
            axes[row_idx, 0].text(
                0.5,
                0.5,
                "No current\nmotion_mask",
                ha="center",
                va="center",
                fontsize=14,
            )
            axes[row_idx, 0].axis("off")

        if nearest_motion_mask is not None:
            nearest_motion_np = nearest_motion_mask.reshape(H, W).detach().cpu().numpy()
            axes[row_idx, 1].imshow(
                nearest_motion_np, cmap="RdYlGn"
            )  # 红色=动态，绿色=静态
            static_nearest = nearest_motion_mask.sum().item()
            axes[row_idx, 1].set_title(
                f"临近视角 motion_mask\n静态点: {static_nearest}/{nearest_motion_mask.numel()} ({100 * static_nearest / nearest_motion_mask.numel():.1f}%)"
            )
            axes[row_idx, 1].axis("off")
        else:
            axes[row_idx, 1].text(
                0.5,
                0.5,
                "No nearest\nmotion_mask",
                ha="center",
                va="center",
                fontsize=14,
            )
            axes[row_idx, 1].axis("off")

        # 组合的 motion_mask（如果两者都存在）
        if current_motion_mask is not None and nearest_motion_mask is not None:
            combined_motion = (
                (current_motion_mask & nearest_motion_mask)
                .reshape(H, W)
                .detach()
                .cpu()
                .numpy()
            )
            axes[row_idx, 2].imshow(combined_motion, cmap="RdYlGn")
            static_combined = (current_motion_mask & nearest_motion_mask).sum().item()
            axes[row_idx, 2].set_title(
                f"组合 motion_mask (交集)\n静态点: {static_combined}/{current_motion_mask.numel()} ({100 * static_combined / current_motion_mask.numel():.1f}%)"
            )
            axes[row_idx, 2].axis("off")
        elif current_motion_mask is not None:
            axes[row_idx, 2].imshow(current_motion_np, cmap="RdYlGn")
            axes[row_idx, 2].set_title("组合 motion_mask\n(只有当前视角)")
            axes[row_idx, 2].axis("off")
        elif nearest_motion_mask is not None:
            axes[row_idx, 2].imshow(nearest_motion_np, cmap="RdYlGn")
            axes[row_idx, 2].set_title("组合 motion_mask\n(只有临近视角)")
            axes[row_idx, 2].axis("off")
        else:
            axes[row_idx, 2].text(
                0.5, 0.5, "No motion_mask", ha="center", va="center", fontsize=14
            )
            axes[row_idx, 2].axis("off")

        # 第三行：最终结果和对比
        row_idx = 2
        axes[row_idx, 0].imshow(d_mask_after_np, cmap="gray")
        valid_after = d_mask_after.sum().item()
        axes[row_idx, 0].set_title(
            f"d_mask (应用motion_mask后)\n有效点: {valid_after}/{d_mask_after.numel()} ({100 * valid_after / d_mask_after.numel():.1f}%)"
        )
        axes[row_idx, 0].axis("off")

        # 最终 d_mask 叠加在真实图像上
        overlay_after = gt_image_np.copy()
        mask_overlay_after = np.zeros_like(overlay_after)
        mask_overlay_after[:, :, 1] = d_mask_after_np  # 绿色通道
        axes[row_idx, 1].imshow(overlay_after * 0.6 + mask_overlay_after * 0.4)
        axes[row_idx, 1].set_title("最终 d_mask 叠加")
        axes[row_idx, 1].axis("off")

        # 差异图：显示被 motion_mask 过滤掉的点
        diff_mask = d_mask_before_np.astype(float) - d_mask_after_np.astype(float)
        diff_img = axes[row_idx, 2].imshow(diff_mask, cmap="Reds", vmin=0, vmax=1)
        filtered_points = (d_mask_before & ~d_mask_after).sum().item()
        axes[row_idx, 2].set_title(
            f"被 motion_mask 过滤的点\n过滤点数: {filtered_points} ({100 * filtered_points / d_mask_before.numel():.1f}%)"
        )
        axes[row_idx, 2].axis("off")
        plt.colorbar(diff_img, ax=axes[row_idx, 2])

        plt.tight_layout()

        # 保存图像
        frame_id = viewpoint_cam.uid if hasattr(viewpoint_cam, "uid") else "unknown"
        save_path = os.path.join(vis_dir, f"mask_combination_frame_{frame_id}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _compute_ncc_loss(
        self,
        viewpoint_cam,
        nearest_cam,
        render_pkg,
        gt_image_gray,
        pixel_noise,
        d_mask,
        weights,
        pixels,
        patch_size,
        sample_num,
        total_patch_size,
        ncc_weight,
    ):
        """
        计算 NCC 损失（内部辅助函数）
        注意：单应性与 NCC 部分必须在 no_grad 外计算，否则多视角损失无法对深度/高斯回传梯度。
        """
        try:
            loss = 0.0
            geo_loss = 0.03 * ((weights * pixel_noise)[d_mask]).mean()
            loss += geo_loss

            # 仅在 no_grad 下做采样与 GT 相关计算，避免对不需要梯度的部分建图
            with torch.no_grad():
                d_mask_flat = d_mask.reshape(-1)
                valid_indices = torch.arange(
                    d_mask_flat.shape[0], device=d_mask.device
                )[d_mask_flat]
                num_valid = valid_indices.shape[0]
                if num_valid > sample_num:
                    rand_indices = torch.randperm(num_valid, device=d_mask.device)[
                        :sample_num
                    ]
                    valid_indices = valid_indices[rand_indices]
                weights_m = weights.reshape(-1)[valid_indices]
                pixels_sampled = pixels.reshape(-1, 2)[valid_indices]
                offsets = self.patch_offsets(patch_size, pixels.device)
                ori_pixels_patch = (
                    pixels_sampled.reshape(-1, 1, 2) / 1.0 + offsets.float()
                )
                H, W = gt_image_gray.squeeze().shape
                pixels_patch = ori_pixels_patch.clone()
                pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                ref_gray_val = F.grid_sample(
                    gt_image_gray.unsqueeze(1),
                    pixels_patch.view(1, -1, 1, 2),
                    align_corners=True,
                )
                ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)
                ref_to_neareast_r = (
                    nearest_cam.R_gt.transpose(-1, -2) @ viewpoint_cam.R_gt
                )
                ref_to_neareast_t = (
                    -ref_to_neareast_r @ viewpoint_cam.T_gt + nearest_cam.T_gt
                )

            # 法向优先用渲染法向；若当前渲染未返回 normal，则用渲染深度估计可微法向。
            normal_map = render_pkg.get("normal", None)
            if normal_map is None:
                normal_map = self._compute_normal_from_rendered_depth(
                    render_pkg["depth"], viewpoint_cam
                )
            if normal_map.dim() == 4:
                normal_map = normal_map.squeeze(0)
            ref_local_n = normal_map.permute(1, 2, 0).reshape(-1, 3)[valid_indices]
            ix, iy = torch.meshgrid(
                torch.arange(W, device=render_pkg["depth"].device),
                torch.arange(H, device=render_pkg["depth"].device),
                indexing="xy",
            )
            rays_d = (
                torch.stack(
                    [
                        (ix - viewpoint_cam.cx / 1) / viewpoint_cam.fx * 1,
                        (iy - viewpoint_cam.cy / 1) / viewpoint_cam.fy * 1,
                        torch.ones_like(ix),
                    ],
                    -1,
                )
                .float()
                .to(render_pkg["depth"].device)
            )
            ref_local_d = render_pkg["depth"] / rays_d[..., 2]
            ref_local_d = ref_local_d.reshape(-1)[valid_indices]

            H_ref_to_neareast = (
                ref_to_neareast_r[None]
                - torch.matmul(
                    ref_to_neareast_t[None, :, None].expand(ref_local_d.shape[0], 3, 1),
                    ref_local_n[:, :, None]
                    .expand(ref_local_d.shape[0], 3, 1)
                    .permute(0, 2, 1),
                )
                / ref_local_d[..., None, None]
            )
            H_ref_to_neareast = torch.matmul(
                nearest_cam.get_k(1)[None].expand(ref_local_d.shape[0], 3, 3),
                H_ref_to_neareast,
            )
            H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(1)

            grid = self.patch_warp(
                H_ref_to_neareast.reshape(-1, 3, 3), ori_pixels_patch
            )
            grid = grid.clone()
            grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
            grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
            _, nearest_image_gray = nearest_cam.get_image()
            sampled_gray_val = F.grid_sample(
                nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True
            )
            sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)

            ncc, ncc_mask = self.lncc(ref_gray_val, sampled_gray_val)
            mask = ncc_mask.reshape(-1)
            ncc = ncc.reshape(-1) * weights_m
            ncc = ncc[mask].squeeze()

            if ncc.numel() > 0:
                ncc_loss = ncc_weight * ncc.mean()
                loss = loss + ncc_loss
            return loss

        except Exception as e:
            print(f"Error in _compute_ncc_loss: {e}")
            return pixel_noise.mean() * 0.0

    # --------------- 8. 地图与优化 ---------------
    def map(
        self,
        stream,
        idx1,
        current_window,
        prune=False,
        iters=1,
        dynamic_network=False,
    ):
        """每轮对当前窗口内多视角累积 loss 再 backward，与 mapper_copy 一致，保证 map 优化效果。

        耗时分析（开启 config mapping.Training.profile_mapping_time=True 可打印各阶段耗时）：
        - render：每轮 num_views 次主渲染，通常占比最高。
        - deform：每视角一次 deform.step。
        - mapping_loss：含 get_loss_mapping、get_depth_order_loss(pair_num=200000)、flatten、flow、multi_view、reg；
          flow 含 2 次 render_flow + generate_flow，multi_view 含 1 次额外 render，均为重头。
        - backward / optim：反传与优化器步进。
        """
        if len(current_window) == 0:
            return
        import numpy as np
        key_opt = []
        if len(current_window) > self.window_size:
            key_opt = self.viewpoints[current_window[0]].keyframe_selection_overlap(
                stream, self.viewpoints, self.viewpoints[current_window[2]].uid
            )
        key_opt = current_window[:3] + key_opt
        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in key_opt]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["mapping"]["Training"]["pose_window"]
        current_window_set = set(key_opt)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        flow_weights = self.config["mapping"]["Training"]["flow_loss"]
        delta = self.config["mapping"]["Training"].get("delta", 5)
        frames_to_optimize = self.config["mapping"]["Training"]["pose_window"]
        from src.utils.Printer import get_msg_prefix

        prefix = get_msg_prefix(FontColor.MAPPER)
        pbar = tqdm(
            range(iters),
            desc=prefix + "Mapping",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            position=0,
            leave=True,
        )
        flatten_w = self.flattening_loss_weight
        visibility_filter_cache = {}
        if getattr(self, "profile_mapping_time", False):
            t_acc = {
                "deform": 0.0,
                "render": 0.0,
                "flow": 0.0,
                "mapping_loss": 0.0,
                "multi_view": 0.0,
                "reg": 0.0,
                "backward": 0.0,
                "optim": 0.0,
            }
            t_count = 0

        for i in pbar:
            if i > 100:
                self.iteration_count += 1
            self.last_sent += 1
            loss_mapping = 0.0
            loss_network = 0.0
            if i < iters / 2:
                dynamic = True
                flow_weights = self.config["mapping"]["Training"]["flow_loss"]
            else:
                dynamic = False
                flow_weights = self.config["mapping"]["Training"].get(
                    "flow_loss_fine", self.config["mapping"]["Training"]["flow_loss"]
                )

            num_views = min(frames_to_optimize, len(viewpoint_stack))
            optimize_window = [vp.uid for vp in viewpoint_stack[:num_views]]
            
            if len(random_viewpoint_stack) >= 2:
                random_kfs = np.random.choice(
                    [vp.uid for vp in random_viewpoint_stack], size=2, replace=False
                ).tolist()
                optimize_window.extend(random_kfs)

            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []
            keyframes_opt = []

            for kf_idx in optimize_window:
                if kf_idx not in self.viewpoints:
                    continue
                viewpoint = self.viewpoints[kf_idx]
                keyframes_opt.append(kf_idx)

                if self.profile_mapping_time:
                    torch.cuda.synchronize()
                    _t0 = time.perf_counter()
                if dynamic_network and self.gaussians.deform_init:
                    dxyz, d_rot, d_scale, d_opac, d_color = self._get_deform_d_values(
                        viewpoint,
                        use_noise=False,
                        detach_outputs=False,
                        iteration=0,
                    )
                else:
                    dxyz = 0
                    d_rot, d_scale, d_opac, d_color = None, 0, None, None
                if self.profile_mapping_time:
                    torch.cuda.synchronize()
                    t_acc["deform"] += time.perf_counter() - _t0
                    _t0 = time.perf_counter()
                
                render_pkg = self._render_viewpoint(
                    viewpoint,
                    dxyz,
                    d_scale,
                    d_rot,
                    d_opac,
                    d_color,
                    return_normal=(flatten_w > 0),
                )
                if self.profile_mapping_time:
                    torch.cuda.synchronize()
                    t_acc["render"] += time.perf_counter() - _t0
                    _t0 = time.perf_counter()
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                    normal,
                ) = self._unpack_render_pkg(render_pkg)
                
                
                if dynamic_network and self.gaussians.deform_init:
                    closest_keyframe = self.find_closest_keyframe(viewpoint.uid)
                    d_value2 = None
                    if (
                        closest_keyframe is not None
                        and closest_keyframe in self.viewpoints
                    ):
                        closest_viewpoint = self.viewpoints[closest_keyframe]
                        flow_loss_delta, d_value2 = self._compute_flow_loss_pair(
                            viewpoint,
                            closest_viewpoint,
                            idx1,
                            dxyz,
                            d_rot,
                            d_scale,
                            flow_weights,
                        )
                        loss_network += flow_loss_delta
                    order_mask = viewpoint.depth > 0
                    loss_order_depth = self.get_depth_order_loss(
                        depth, viewpoint.depth, order_mask
                    )
                    l = (
                        get_loss_mapping(
                            self.config["mapping"],
                            image,
                            depth,
                            viewpoint,
                            opacity,
                            rm_dynamic=not dynamic_network,
                            dynamic=dynamic,
                        )
                        + 0.1 * loss_order_depth
                    )
                    loss_mapping += l
                    loss_network += l
                    
                    # if flatten_w > 0 and normal is not None:
                    #     loss_mapping += flatten_w * self.get_loss_flattening(
                    #         normal, viewpoint, opacity=opacity
                    #     )
                    if (
                        closest_keyframe is not None
                        and closest_keyframe in self.viewpoints
                        and not dynamic
                    ):
                        loss_mapping += self.compute_multi_view_loss(
                            viewpoint,
                            render_pkg,
                            self.gaussians,
                            self.pipeline_params,
                            self.background,
                            d_value2["d_xyz"],
                            d_value2["d_scaling"],
                            d_value2["d_rotation"],
                            d_value2["d_opacity"],
                            d_value2["d_color"],
                            self.viewpoints[closest_keyframe],
                        )
                    loss_network += self._add_deform_regularization_losses(
                        viewpoint, delta=delta
                    )
                
                if self.profile_mapping_time:
                    torch.cuda.synchronize()
                    t_acc["mapping_loss"] += time.perf_counter() - _t0
                
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)
                del render_pkg

            if len(viewspace_point_tensor_acm) == 0:
                continue

            if self.profile_mapping_time:
                torch.cuda.synchronize()
                _t0 = time.perf_counter()
            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            combined_vis = visibility_filter_acm[0].clone()
            for v in visibility_filter_acm[1:]:
                combined_vis = combined_vis | v
            # loss_mapping += self.scale_loss_weight * self.get_loss_flat(combined_vis)
            # 与 mapper_copy 一致：mapping 与 network 分开反传，deform 只受 loss_network 更新，避免重建梯度干扰动态优化
            loss_mapping.backward(retain_graph=True)
            if self.profile_mapping_time:
                torch.cuda.synchronize()
                t_acc["backward"] += time.perf_counter() - _t0
                _t0 = time.perf_counter()

            gaussian_split = False
            with torch.no_grad():
                for idx in range(len(keyframes_opt)):
                    kf_idx = keyframes_opt[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched_acm[idx] > 0).long()
                    visibility_filter_cache[kf_idx] = visibility_filter_acm[
                        idx
                    ].detach()
                for idx in range(len(viewspace_point_tensor_acm)):
                    vf = visibility_filter_acm[idx]
                    self.gaussians.max_radii2D[vf] = torch.max(
                        self.gaussians.max_radii2D[vf],
                        radii_acm[idx][vf],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx].detach(), vf
                    )
                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                    and i > 100
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True
                if (
                    (self.iteration_count % self.gaussian_reset) == 0
                    and (not update_gaussian)
                    and i > 100
                ):
                    self.printer.print(
                        "Resetting the opacity of non-visible Gaussians",
                        FontColor.MAPPER,
                    )
                    visibility_filter_for_reset = [
                        visibility_filter_cache[k]
                        for k in current_window
                        if k in visibility_filter_cache
                    ]
                    if len(visibility_filter_for_reset) > 0:
                        self.gaussians.reset_opacity_nonvisible(
                            visibility_filter_for_reset
                        )
                    gaussian_split = True
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                for kf in keyframes_opt[:num_views]:
                    if kf not in self.viewpoints:
                        continue
                    vp = self.viewpoints[kf]
                    if vp.uid == 0:
                        continue
                    update_pose(vp)
            if (
                dynamic_network
                and self.gaussians.deform_init
                and isinstance(loss_network, torch.Tensor)
                and loss_network.requires_grad
            ):
                loss_network.backward()
            with torch.no_grad():
                if dynamic_network and self.gaussians.deform_init:
                    self.gaussians.deform.optimizer.step()
                    self.gaussians.deform.optimizer.zero_grad(set_to_none=True)
                if i > 100:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    self.gaussians.update_learning_rate(self.iteration_count)
                else:
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
            if self.profile_mapping_time:
                torch.cuda.synchronize()
                t_acc["optim"] += time.perf_counter() - _t0
                t_count += 1
                if t_count > 0 and t_count % 100 == 0:
                    n = t_count
                    self.printer.print(
                        f"[Map 耗时] 近{n}轮均: deform={t_acc['deform'] / n:.3f}s render={t_acc['render'] / n:.3f}s "
                        f"loss_compute={t_acc['mapping_loss'] / n:.3f}s backward={t_acc['backward'] / n:.3f}s optim={t_acc['optim'] / n:.3f}s",
                        FontColor.MAPPER,
                    )

        # Prune：需要 current_window 可见性，先补全再 prune
        if prune and len(current_window) == self.window_size:
            with torch.no_grad():
                for kf_idx in current_window:
                    if kf_idx in self.occ_aware_visibility:
                        continue
                    vp = self.viewpoints.get(kf_idx)
                    if vp is None:
                        continue
                    render_pkg = self._render_viewpoint(
                        vp, 0, 0, None, None, None, return_normal=False
                    )
                    _, _, visibility_filter, _, _, _, n_touched, _ = (
                        self._unpack_render_pkg(render_pkg)
                    )
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()
                    visibility_filter_cache[kf_idx] = visibility_filter.detach()
            prune_mode = self.config["mapping"]["Training"]["prune_mode"]
            prune_coviz = 3
            self.gaussians.n_obs.fill_(0)
            target_numel = self.gaussians.n_obs.numel()
            for window_idx, visibility in self.occ_aware_visibility.items():
                aligned_visibility = self._align_visibility_mask(
                    visibility, target_numel
                ).long()
                self.gaussians.n_obs += aligned_visibility.cpu()
                self.occ_aware_visibility[window_idx] = aligned_visibility
            to_prune = None
            if prune_mode == "odometry":
                to_prune = self.gaussians.n_obs < 3
            if prune_mode == "slam":
                sorted_window = sorted(current_window, reverse=True)
                mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                if not self.initialized:
                    mask = self.gaussians.unique_kfIDs >= 0
                to_prune = torch.logical_and(
                    self.gaussians.n_obs <= prune_coviz,
                    mask,
                )
            if to_prune is not None and self.monocular:
                self.gaussians.prune_points(to_prune.cuda())
                for idx in range(len(current_window)):
                    current_idx = current_window[idx]
                    if current_idx in self.occ_aware_visibility:
                        self.occ_aware_visibility[current_idx] = (
                            self.occ_aware_visibility[current_idx][~to_prune]
                        )
            if not self.initialized:
                self.initialized = True
            return False

        if getattr(self, "profile_mapping_time", False) and t_count > 0:
            n = t_count
            self.printer.print(
                f"[Map 耗时汇总] 共{n}轮 总: deform={t_acc['deform']:.1f}s render={t_acc['render']:.1f}s "
                f"loss_compute={t_acc['mapping_loss']:.1f}s backward={t_acc['backward']:.1f}s optim={t_acc['optim']:.1f}s | "
                f"占比: render={100 * t_acc['render'] / (t_acc['deform'] + t_acc['render'] + t_acc['mapping_loss'] + t_acc['backward'] + t_acc['optim']):.0f}%",
                FontColor.MAPPER,
            )

        if self.online_plotting:
            from thirdparty.gaussian_splatting.utils.image_utils import psnr
            from src.utils.eval_utils import plot_rgbd_silhouette
            import cv2
            import numpy as np

            cur_idx = current_window[np.array(current_window).argmax()]
            viewpoint = self.viewpoints[cur_idx]
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                depth,
            ) = (
                render_pkg["render"].detach(),
                render_pkg["depth"].detach(),
            )
            gt_image = viewpoint.original_image
            gt_depth = viewpoint.depth

            if (
                viewpoint.uid != self.video_idxs[0]
            ):  # first mapping frame is reference for exposure
                image = (
                    torch.exp(viewpoint.exposure_a.detach())
                ) * image + viewpoint.exposure_b.detach()

            image = torch.clamp(image, 0.0, 1.0)
            gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)

            pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
                np.uint8
            )
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
            mask = gt_image > 0
            psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
            diff_depth_l1 = torch.abs(depth.detach().cpu() - gt_depth)
            diff_depth_l1 = diff_depth_l1 * (gt_depth > 0)
            depth_l1 = diff_depth_l1.sum() / (gt_depth > 0).sum()

            # Add plotting 2x3 grid here
            plot_dir = self.save_dir + "/online_plots"
            plot_rgbd_silhouette(
                gt_image,
                gt_depth,
                image,
                depth,
                diff_depth_l1,
                psnr_score.item(),
                depth_l1,
                plot_dir=plot_dir,
                idx=str(cur_idx),
                diff_rgb=np.abs(gt - pred),
            )

        return gaussian_split

    def visualize_flow(self, flow):
        flow_uv = flow[..., :2].copy() * 5.0  # 放大5倍使弱运动可见

        # --- 运动方向与速度的HSV编码 ---
        magnitude = np.linalg.norm(flow_uv, axis=-1)  # 速度幅值
        angle = np.arctan2(flow_uv[..., 1], flow_uv[..., 0])  # 运动方向 [-π, π]

        # 构建HSV图像
        hsv = np.zeros((flow_uv.shape[0], flow_uv.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = (angle + np.pi) / (2 * np.pi) * 180  # 色相：方向[0°, 180°]
        hsv[..., 1] = cv2.normalize(
            magnitude, None, 0, 255, cv2.NORM_MINMAX
        )  # 饱和度：速度
        hsv[..., 2] = 255  # 亮度：固定最大值

        # 转换并保存为BGR（修复：补全字符串和变量名）
        flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return flow_img

    def final_refine(self, prune=False, iters=26000):
        """步骤1：导入指标、初始化 PSNR/SSIM/LPIPS 记录与输出目录。"""
        self.printer.print("Starting final refinement", FontColor.MAPPER)
        from thirdparty.gaussian_splatting.utils.image_utils import psnr
        from thirdparty.gaussian_splatting.utils.loss_utils import ssim
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        psnr_history, ssim_history, lpips_history, iteration_history = [], [], [], []
        cal_lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to("cuda")
        refine_output_dir = os.path.join(self.save_dir, "final_refine_visualizations")
        os.makedirs(refine_output_dir, exist_ok=True)
        depth_output_dir = os.path.join(refine_output_dir, "depth")
        normal_output_dir = os.path.join(refine_output_dir, "normal")
        flow_output_dir = os.path.join(refine_output_dir, "flow")
        for d in (depth_output_dir, normal_output_dir, flow_output_dir):
            os.makedirs(d, exist_ok=True)
        # 步骤2：预更新所有关键帧位姿与深度（get_w2c_and_depth → update_RT/depth，可选 update_mapping_points）
        for keyframe_idx, frame_idx in zip(self.video_idxs, self.keyframe_idxs):
            _, _, depth_gtd, _, motion_mask, normal, mono, static_msk = (
                self.frame_reader[frame_idx]
            )
            depth_gt_numpy = depth_gtd.cpu().numpy()

            intrinsics = as_intrinsics_matrix(self.frame_reader.get_intrinsic()).to(
                self.device
            )

            mono_depth = load_mono_depth(frame_idx, self.save_dir).to(self.device)

            depth_temp, w2c_temp, invalid = self.get_w2c_and_depth(
                keyframe_idx,
                frame_idx,
                mono_depth,
                motion_mask,
                depth_gt_numpy,
                normal,
                mono,
                static_msk,
                init=False,
            )

            w2c_old = torch.cat(
                (
                    self.cameras[keyframe_idx].R,
                    self.cameras[keyframe_idx].T.unsqueeze(-1),
                ),
                dim=1,
            )
            w2c_old = torch.cat(
                (w2c_old, torch.tensor([[0, 0, 0, 1]], device="cuda")), dim=0
            )

            self.cameras[keyframe_idx].update_RT(w2c_temp[:3, :3], w2c_temp[:3, 3])

            self.cameras[keyframe_idx].depth = depth_temp.cpu().numpy()

            if keyframe_idx in self.viewpoints:
                self.viewpoints[keyframe_idx].update_RT(
                    w2c_temp[:3, :3], w2c_temp[:3, 3]
                )

                self.viewpoints[keyframe_idx].depth = depth_temp.cpu().numpy()

            if self.move_points and self.is_kf[keyframe_idx]:
                if invalid:
                    self.update_mapping_points(
                        keyframe_idx,
                        w2c_temp,
                        w2c_old,
                        depth_temp,
                        self.depth_dict[keyframe_idx],
                        intrinsics,
                        method="rigid",
                    )
                else:
                    self.update_mapping_points(
                        keyframe_idx,
                        w2c_temp,
                        w2c_old,
                        depth_temp,
                        self.depth_dict[keyframe_idx],
                        intrinsics,
                    )

                    self.depth_dict[keyframe_idx] = depth_temp

        random_viewpoint_stack = list(self.viewpoints.values())
        # 步骤3：迭代 iters 轮——随机采样视角、变形+渲染、RGB/深度/正则损失、反传与 optimizer.step
        from src.utils.Printer import get_msg_prefix

        prefix = get_msg_prefix(FontColor.MAPPER)
        pbar = tqdm(
            range(iters),
            desc=prefix + "Final refinement",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            position=0,
            leave=True,
        )
        flatten_w = self.flattening_loss_weight
        for iteration in pbar:
            loss = 0
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []
            for _ in range(10):
                # 随机选择视角进行渲染
                rand_idx = np.random.randint(0, len(random_viewpoint_stack))
                viewpoint = random_viewpoint_stack[rand_idx]
                # print("self.dynamic_model", self.dynamic_model)
                print("self.gaussians.deform_init", self.gaussians.deform_init)
                if self.dynamic_model and self.gaussians.deform_init:
                    dxyz, d_rot, d_scale, d_opac, d_color = self._get_deform_d_values(
                        viewpoint,
                        use_noise=False,
                        detach_outputs=False,
                        iteration=0,
                    )
                    closest_keyframe = self.find_closest_keyframe(viewpoint.uid)
                    if (
                        closest_keyframe is not None
                        and closest_keyframe in self.viewpoints
                    ):
                        d_value2 = self._get_deform_d_values_for_fid(
                            self.viewpoints[closest_keyframe].fid,
                            self.viewpoints[closest_keyframe].camera_center,
                            iteration=0,
                        )
                        d_xyz2 = d_value2["d_xyz"]
                        d_rot2 = d_value2["d_rotation"]
                        d_scale2 = d_value2["d_scaling"]
                        d_opac2 = d_value2.get("d_opacity")
                        d_color2 = d_value2.get("d_color")
                    else:
                        d_value2 = None
                        d_xyz2, d_rot2, d_scale2, d_opac2, d_color2 = (
                            0,
                            0,
                            0,
                            None,
                            None,
                        )
                else:
                    dxyz, d_rot, d_scale, d_opac, d_color = 0, 0, 0, None, None
                    d_xyz2, d_rot2, d_scale2, d_opac2, d_color2 = 0, 0, 0, None, None
                    d_value2 = None

                render_pkg = self._render_viewpoint(
                    viewpoint,
                    dxyz,
                    d_scale,
                    d_rot,
                    d_opac,
                    d_color,
                    return_normal=(flatten_w > 0),
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                    normal,
                ) = self._unpack_render_pkg(render_pkg)

                image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
                gt_image = viewpoint.original_image.cuda()
                gt_depth = torch.from_numpy(viewpoint.depth).to(
                    dtype=torch.float32, device=image.device
                )[None]
                depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
                # if flatten_w > 0 and normal is not None:
                #     loss += flatten_w * self.get_loss_flattening(
                #         normal, viewpoint, opacity=opacity
                #     )
                if self.dynamic_model:
                    Ll1 = l1_loss(image, gt_image)
                    loss += (1.0 - self.opt_params.lambda_dssim) * (
                        Ll1
                    ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
                    loss += 1e-4 * self.gaussians.deform.deform.arap_loss(
                        t=viewpoint.fid,
                        delta_t=5 * self.gaussians.time_interval,
                        t_samp_num=8,
                    )
                else:
                    Ll1 = l1_loss(image, gt_image, mask=viewpoint.motion_mask)
                    loss += (1.0 - self.opt_params.lambda_dssim) * (
                        Ll1
                    ) + self.opt_params.lambda_dssim * (
                        1.0 - ssim(image, gt_image, mask=viewpoint.motion_mask)
                    )
                    depth_pixel_mask = (
                        viewpoint.motion_mask.view(*gt_depth.shape) * depth_pixel_mask
                    )

                l1_depth = torch.abs(
                    depth * depth_pixel_mask - gt_depth * depth_pixel_mask
                )
                loss += 0.1 * l1_depth.mean()
                loss_depth = depth_loss_dpt(depth, viewpoint.depth)
                order_mask = viewpoint.depth > 0
                loss_order_depth = self.get_depth_order_loss(
                    depth, viewpoint.depth, order_mask
                )
                loss += 0.1 * loss_order_depth  # 1e-3 * loss_depth +
                # 仅当开启 save_debug_visualization 且满足迭代条件时才绘制 d_mask / mask_combination
                if (
                    getattr(self, "save_debug_visualization", False)
                    and iteration % 400 == 0
                    and iteration > 7000
                ):
                    self.visualize_mask = True
                    self.visualize_mask_combination = True
                else:
                    self.visualize_mask = False
                    self.visualize_mask_combination = False
                # if iteration > 7000:
                #     if closest_keyframe is not None and closest_keyframe in self.viewpoints:
                #         loss += self.compute_multi_view_loss(viewpoint, render_pkg, self.gaussians,
                #                                              self.pipeline_params, self.background,
                #                                              d_xyz2, d_scale2, d_rot2, d_opac2,
                #                                              d_color2,
                #                                              self.viewpoints[closest_keyframe])

                viewspace_point_tensor_acm.append(viewspace_point_tensor.detach())
                visibility_filter_acm.append(visibility_filter.detach())
                radii_acm.append(radii.detach())
                n_touched_acm.append(n_touched.detach())

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss += 10 * isotropic_loss.mean()
            # if flatten_w > 0 and self.scale_loss_weight > 0:
            #     loss += self.scale_loss_weight * self.get_loss_flat(visibility_filter)

            loss.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )

                self.gaussians.optimizer.step()

                self.gaussians.optimizer.zero_grad(set_to_none=True)

                self.gaussians.update_learning_rate(self.iteration_count)

                self.keyframe_optimizers.step()

                self.keyframe_optimizers.zero_grad(set_to_none=True)
                if self.dynamic_model and self.gaussians.deform_init:
                    self.gaussians.deform.optimizer.step()
                    self.gaussians.deform.optimizer.zero_grad(set_to_none=True)

                # 步骤4：每 1000 迭代计算并记录 PSNR/SSIM/LPIPS（遍历关键帧取平均）
                if iteration % 1000 == 0 and iteration > 0:
                    try:
                        psnr_list, ssim_list, lpips_list = [], [], []

                        # 遍历所有关键帧（参考eval_utils.py）
                        for kf_idx, video_idx in zip(
                            self.keyframe_idxs, self.video_idxs
                        ):
                            frame = self.cameras[video_idx]
                            (
                                _,
                                gt_img,
                                gt_depth_data,
                                _,
                                motion_msk,
                                normal_data,
                                mono_data,
                                static_mask,
                            ) = self.frame_reader[kf_idx]
                            gt_img = gt_img.squeeze().cuda()
                            gt_depth = gt_depth_data.cpu().numpy()
                            if self.dynamic_model and self.gaussians.deform_init:
                                dxyz, d_rot, d_scale, d_opac, d_color = (
                                    self._get_deform_d_values(
                                        frame,
                                        use_noise=False,
                                        detach_outputs=False,
                                        iteration=0,
                                    )
                                )
                            else:
                                dxyz, d_rot, d_scale, d_opac, d_color = (
                                    0,
                                    0,
                                    0,
                                    None,
                                    None,
                                )
                            render_pkg = self._render_viewpoint(
                                frame, dxyz, d_scale, d_rot, d_opac, d_color
                            )
                            rendered_img = render_pkg["render"].detach()

                            # 应用曝光补偿（如果不是第一帧）
                            if video_idx > 0:
                                rendered_img = (
                                    torch.exp(frame.exposure_a.detach())
                                ) * rendered_img + frame.exposure_b.detach()
                            rendered_img = torch.clamp(rendered_img, 0.0, 1.0)

                            # 准备mask（参考eval_utils.py的逻辑）
                            mask = gt_img > 0
                            depth_mask = gt_depth > 0
                            # 根据模型类型调整mask
                            if not self.gaussians.deform_init:
                                # print("eval remove motion region")
                                mask = (
                                    mask
                                    * motion_mask.view(*depth_mask.shape)
                                    * torch.from_numpy(depth_mask).to(
                                        device=motion_mask.device
                                    )
                                )
                                depth_mask = (
                                    depth_mask
                                    * motion_mask.view(*depth_mask.shape).cpu().numpy()
                                )
                            else:
                                mask = mask * torch.from_numpy(depth_mask).to(
                                    device=motion_mask.device
                                )

                            # 计算PSNR（使用masked区域）
                            psnr_value = psnr(
                                (rendered_img[mask]).unsqueeze(0),
                                (gt_img[mask]).unsqueeze(0),
                            )

                            # 计算SSIM（使用全图）
                            ssim_value = ssim(
                                rendered_img.unsqueeze(0), gt_img.unsqueeze(0)
                            )

                            # 计算LPIPS（使用全图）
                            lpips_value = cal_lpips(
                                rendered_img.unsqueeze(0), gt_img.unsqueeze(0)
                            )

                            psnr_list.append(psnr_value.item())
                            ssim_list.append(ssim_value.item())
                            lpips_list.append(lpips_value.item())

                        # 计算所有关键帧的平均值
                        avg_psnr = np.mean(psnr_list)
                        avg_ssim = np.mean(ssim_list)
                        avg_lpips = np.mean(lpips_list)

                        psnr_history.append(avg_psnr)
                        ssim_history.append(avg_ssim)
                        lpips_history.append(avg_lpips)
                        iteration_history.append(iteration)

                        # 打印平均评价指标
                        print(
                            f"\nIteration {iteration} (avg over {len(psnr_list)} frames): "
                            f"PSNR = {avg_psnr:.2f} dB, SSIM = {avg_ssim:.4f}, LPIPS = {avg_lpips:.4f}"
                        )
                    except Exception as e:
                        print(
                            f"\nWarning: Failed to calculate metrics at iteration {iteration}: {e}"
                        )

                # 步骤5：每 1000 迭代保存深度/法线/光流可视化（仅当 save_debug_visualization 为 True）
                if (
                    getattr(self, "save_debug_visualization", False)
                    and iteration % 1000 == 0
                    and iteration > 0
                ):
                    try:
                        print(f"\nSaving visualizations for iteration {iteration}...")
                        for kf_idx, video_idx in zip(
                            self.keyframe_idxs, self.video_idxs
                        ):
                            frame = self.cameras[video_idx]
                            if self.dynamic_model and self.gaussians.deform_init:
                                dxyz, d_rot, d_scale, d_opac, d_color = (
                                    self._get_deform_d_values(
                                        frame,
                                        use_noise=False,
                                        detach_outputs=False,
                                        iteration=0,
                                    )
                                )
                            else:
                                dxyz, d_rot, d_scale, d_opac, d_color = (
                                    0,
                                    0,
                                    0,
                                    None,
                                    None,
                                )
                            render_pkg = self._render_viewpoint(
                                frame, dxyz, d_scale, d_rot, d_opac, d_color
                            )
                            rendered_depth = render_pkg["depth"]
                            depth_vis = rendered_depth[0].detach().cpu().numpy()
                            depth_vis = (depth_vis - depth_vis.min()) / (
                                depth_vis.max() - depth_vis.min() + 1e-8
                            )
                            depth_colored = plt.cm.turbo(depth_vis)[:, :, :3]
                            depth_img = (depth_colored * 255).astype(np.uint8)
                            depth_filename = (
                                f"iter_{iteration:06d}_frame_{video_idx:04d}.png"
                            )
                            cv2.imwrite(
                                os.path.join(depth_output_dir, depth_filename),
                                cv2.cvtColor(depth_img, cv2.COLOR_RGB2BGR),
                            )
                            depth_filter = torch.nn.functional.avg_pool2d(
                                rendered_depth.unsqueeze(0),
                                kernel_size=3,
                                stride=1,
                                padding=1,
                            )
                            depth_filter = torch.nn.functional.avg_pool2d(
                                depth_filter, kernel_size=3, stride=1, padding=1
                            )
                            focal_x = torch.Tensor([frame.fx]).cuda()
                            focal_y = torch.Tensor([frame.fy]).cuda()
                            xyz = self.depth_to_xyz(depth_filter, focal_x, focal_y)
                            xyz_i = xyz[0, :][None, :, :, :]
                            normal_map = self.get_surface_normalv2(xyz_i).permute(
                                (3, 2, 0, 1)
                            )
                            normal_vis = normal_map.squeeze().detach().cpu().numpy()
                            if normal_vis.shape[0] == 3:
                                normal_vis = np.transpose(normal_vis, (1, 2, 0))
                            normal_vis = (normal_vis + 1) / 2
                            normal_img = (normal_vis * 255).astype(np.uint8)
                            normal_filename = (
                                f"iter_{iteration:06d}_frame_{video_idx:04d}.png"
                            )
                            cv2.imwrite(
                                os.path.join(normal_output_dir, normal_filename),
                                cv2.cvtColor(normal_img, cv2.COLOR_RGB2BGR),
                            )
                            if self.dynamic_model and self.gaussians.deform_init:
                                try:
                                    closest_keyframe = self.find_closest_keyframe(
                                        frame.uid
                                    )
                                    if (
                                        closest_keyframe is not None
                                        and closest_keyframe in self.viewpoints
                                    ):
                                        closest_viewpoint = self.viewpoints[
                                            closest_keyframe
                                        ]
                                        d_xyz1, d_rot1, d_scale1, _, _ = (
                                            self._get_deform_d_values(
                                                frame,
                                                use_noise=False,
                                                detach_outputs=False,
                                                iteration=0,
                                            )
                                        )
                                        d_value2 = self._get_deform_d_values_for_fid(
                                            closest_viewpoint.fid,
                                            closest_viewpoint.camera_center,
                                            iteration=0,
                                        )
                                        d_xyz2, d_rot2, d_scale2 = (
                                            d_value2["d_xyz"],
                                            d_value2["d_rotation"],
                                            d_value2["d_scaling"],
                                        )
                                        flow_pkg = render_flow(
                                            pc=self.gaussians,
                                            viewpoint_camera1=frame,
                                            viewpoint_camera2=closest_viewpoint,
                                            d_xyz1=d_xyz1,
                                            d_xyz2=d_xyz2,
                                            d_rotation1=d_rot1,
                                            d_scaling1=d_scale1,
                                        )
                                        if "render" in flow_pkg:
                                            flow = (
                                                flow_pkg["render"]
                                                .detach()
                                                .cpu()
                                                .numpy()
                                            )
                                            flow = np.transpose(flow, (1, 2, 0))
                                            flow_vis = self.visualize_flow(flow)
                                            flow_filename = f"iter_{iteration:06d}_frame_{video_idx:04d}.png"
                                            cv2.imwrite(
                                                os.path.join(
                                                    flow_output_dir, flow_filename
                                                ),
                                                flow_vis,
                                            )
                                except Exception as e:
                                    print(
                                        f"\nWarning: Failed to save flow for frame {video_idx} at iteration {iteration}: {e}"
                                    )
                        print(
                            f"Saved visualizations for {len(self.keyframe_idxs)} frames at iteration {iteration}"
                        )
                    except Exception as e:
                        print(
                            f"\nWarning: Failed to save visualizations at iteration {iteration}: {e}"
                        )

        # 步骤6：若有指标记录则画 PSNR/SSIM/LPIPS 曲线并保存，写 metrics_data.txt 与 metrics_summary.txt
        if len(psnr_history) > 0 and len(ssim_history) > 0 and len(lpips_history) > 0:
            if not (
                len(psnr_history)
                == len(ssim_history)
                == len(lpips_history)
                == len(iteration_history)
            ):
                print(f"⚠️ Warning: Metrics history length mismatch!")
                print(
                    f"   psnr: {len(psnr_history)}, ssim: {len(ssim_history)}, "
                    f"lpips: {len(lpips_history)}, iteration: {len(iteration_history)}"
                )
                # 截断到最短长度
                min_len = min(
                    len(psnr_history),
                    len(ssim_history),
                    len(lpips_history),
                    len(iteration_history),
                )
                psnr_history = psnr_history[:min_len]
                ssim_history = ssim_history[:min_len]
                lpips_history = lpips_history[:min_len]
                iteration_history = iteration_history[:min_len]
                print(f"   Truncated to {min_len} samples")

            try:
                fig, ax1 = plt.subplots(figsize=(14, 8))

                # 左Y轴：PSNR
                color_psnr = "tab:blue"
                ax1.set_xlabel("Iteration", fontsize=13, fontweight="bold")
                ax1.set_ylabel("PSNR", color=color_psnr, fontsize=13, fontweight="bold")
                line1 = ax1.plot(
                    iteration_history,
                    psnr_history,
                    color=color_psnr,
                    linewidth=2.5,
                    marker="o",
                    markersize=5,
                    label="PSNR",
                    alpha=0.8,
                )
                ax1.tick_params(axis="y", labelcolor=color_psnr, labelsize=11)
                ax1.grid(True, alpha=0.3, linestyle="--")

                # 右Y轴：SSIM 和 LPIPS
                ax2 = ax1.twinx()
                color_ssim = "tab:green"
                color_lpips = "tab:red"
                ax2.set_ylabel("SSIM / LPIPS", fontsize=13, fontweight="bold")

                line2 = ax2.plot(
                    iteration_history,
                    ssim_history,
                    color=color_ssim,
                    linewidth=2.5,
                    marker="s",
                    markersize=5,
                    label="SSIM",
                    alpha=0.8,
                )
                line3 = ax2.plot(
                    iteration_history,
                    lpips_history,
                    color=color_lpips,
                    linewidth=2.5,
                    marker="^",
                    markersize=5,
                    label="LPIPS",
                    alpha=0.8,
                )
                ax2.tick_params(axis="y", labelsize=11)

                # 合并图例
                lines = line1 + line2 + line3
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc="best", fontsize=12, framealpha=0.9)

                # 设置标题
                plt.title(
                    "Metrics during Final Refinement",
                    fontsize=15,
                    fontweight="bold",
                    pad=20,
                )
                plt.tight_layout()

                # 保存综合曲线图
                metrics_plot_path = os.path.join(refine_output_dir, "metrics_curve.png")
                plt.savefig(metrics_plot_path, dpi=150, bbox_inches="tight")
                plt.close()
                # 创建包含三个子图的图表
                fig, axes = plt.subplots(3, 1, figsize=(12, 14))

                # ========== 保存单独的PSNR曲线图 ==========
                plt.figure(figsize=(12, 6))
                plt.plot(
                    iteration_history,
                    psnr_history,
                    "b-",
                    linewidth=2,
                    marker="o",
                    markersize=4,
                )
                plt.xlabel("Iteration", fontsize=12)
                plt.ylabel("PSNR (dB)", fontsize=12)
                plt.title(
                    "PSNR during Final Refinement (Higher is Better)",
                    fontsize=14,
                    fontweight="bold",
                )
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                psnr_plot_path = os.path.join(refine_output_dir, "psnr_curve.png")
                plt.savefig(psnr_plot_path, dpi=150, bbox_inches="tight")
                plt.close()

                # ========== 保存单独的SSIM曲线图 ==========
                plt.figure(figsize=(12, 6))
                plt.plot(
                    iteration_history,
                    ssim_history,
                    "g-",
                    linewidth=2,
                    marker="s",
                    markersize=4,
                )
                plt.xlabel("Iteration", fontsize=12)
                plt.ylabel("SSIM", fontsize=12)
                plt.title(
                    "SSIM during Final Refinement (Higher is Better)",
                    fontsize=14,
                    fontweight="bold",
                )
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                ssim_plot_path = os.path.join(refine_output_dir, "ssim_curve.png")
                plt.savefig(ssim_plot_path, dpi=150, bbox_inches="tight")
                plt.close()

                # ========== 保存单独的LPIPS曲线图 ==========
                plt.figure(figsize=(12, 6))
                plt.plot(
                    iteration_history,
                    lpips_history,
                    "r-",
                    linewidth=2,
                    marker="^",
                    markersize=4,
                )
                plt.xlabel("Iteration", fontsize=12)
                plt.ylabel("LPIPS", fontsize=12)
                plt.title(
                    "LPIPS during Final Refinement (Lower is Better)",
                    fontsize=14,
                    fontweight="bold",
                )
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                lpips_plot_path = os.path.join(refine_output_dir, "lpips_curve.png")
                plt.savefig(lpips_plot_path, dpi=150, bbox_inches="tight")
                plt.close()

                # 保存评价指标数据到文本文件（添加编码）
                metrics_data_path = os.path.join(refine_output_dir, "metrics_data.txt")
                with open(metrics_data_path, "w", encoding="utf-8") as f:
                    f.write("Iteration\tPSNR(dB)\tSSIM\tLPIPS\n")
                    for iter_num, psnr_val, ssim_val, lpips_val in zip(
                        iteration_history, psnr_history, ssim_history, lpips_history
                    ):
                        f.write(
                            f"{iter_num}\t{psnr_val:.4f}\t{ssim_val:.4f}\t{lpips_val:.4f}\n"
                        )

                # 保存统计摘要（添加编码）
                summary_path = os.path.join(refine_output_dir, "metrics_summary.txt")
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write("=" * 60 + "\n")
                    f.write("Final Refinement Metrics Summary\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Total Iterations: {iters}\n")
                    f.write(f"Recorded Samples: {len(psnr_history)}\n\n")
                    f.write("-" * 60 + "\n")
                    f.write("PSNR (Peak Signal-to-Noise Ratio) - Higher is Better\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"  Final: {psnr_history[-1]:.2f} dB\n")
                    f.write(f"  Max:   {max(psnr_history):.2f} dB\n")
                    f.write(f"  Mean:  {np.mean(psnr_history):.2f} dB\n")
                    f.write(f"  Min:   {min(psnr_history):.2f} dB\n")
                    f.write(f"  Std:   {np.std(psnr_history):.2f} dB\n\n")
                    f.write("-" * 60 + "\n")
                    f.write("SSIM (Structural Similarity Index) - Higher is Better\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"  Final: {ssim_history[-1]:.4f}\n")
                    f.write(f"  Max:   {max(ssim_history):.4f}\n")
                    f.write(f"  Mean:  {np.mean(ssim_history):.4f}\n")
                    f.write(f"  Min:   {min(ssim_history):.4f}\n")
                    f.write(f"  Std:   {np.std(ssim_history):.4f}\n\n")
                    f.write("-" * 60 + "\n")
                    f.write(
                        "LPIPS (Learned Perceptual Image Patch Similarity) - Lower is Better\n"
                    )
                    f.write("-" * 60 + "\n")
                    f.write(f"  Final: {lpips_history[-1]:.4f}\n")
                    f.write(f"  Min:   {min(lpips_history):.4f}\n")
                    f.write(f"  Mean:  {np.mean(lpips_history):.4f}\n")
                    f.write(f"  Max:   {max(lpips_history):.4f}\n")
                    f.write(f"  Std:   {np.std(lpips_history):.4f}\n\n")
                    f.write("=" * 60 + "\n")

                print(f"\n{'=' * 60}")
                print("Final Refinement Metrics Summary")
                print("=" * 60)
                print(f"Metrics curve saved to: {metrics_plot_path}")
                print(f"PSNR curve saved to: {psnr_plot_path}")
                print(f"Metrics data saved to: {metrics_data_path}")
                print(f"Summary saved to: {summary_path}")
                print(
                    f"\nPSNR  - Final: {psnr_history[-1]:.2f} dB, Max: {max(psnr_history):.2f} dB, Mean: {np.mean(psnr_history):.2f} dB"
                )
                print(
                    f"SSIM  - Final: {ssim_history[-1]:.4f}, Max: {max(ssim_history):.4f}, Mean: {np.mean(ssim_history):.4f}"
                )
                print(
                    f"LPIPS - Final: {lpips_history[-1]:.4f}, Min: {min(lpips_history):.4f}, Mean: {np.mean(lpips_history):.4f}"
                )
                print("=" * 60)

            except Exception as e:
                print(f"⚠️ Error generating metrics visualization: {e}")
                import traceback

                traceback.print_exc()
        else:
            print(
                f"⚠️ Warning: Insufficient metrics data collected (only {len(psnr_history)} samples)"
            )
            print(
                f"   Note: Metrics are collected every 100 iterations, total iterations: {iters}"
            )

        self.printer.print("Final refinement done", FontColor.MAPPER)

    def get_loss_flat(self, visibility_filter=None):
        """PGSR 风格扁平约束：L_flat = mean_i min(sx_i, sy_i, sz_i)，可选仅在可见高斯上统计。

        visibility_filter: (N,) bool，可见高斯掩码；若为 None，则在所有高斯上求平均。
        """
        scaling = self.gaussians.get_scaling  # (N, 3)
        if scaling.shape[0] == 0:
            return scaling.new_tensor(0.0)
        if (
            visibility_filter is not None
            and visibility_filter.shape[0] == scaling.shape[0]
        ):
            scaling = scaling[visibility_filter]
            if scaling.shape[0] == 0:
                return scaling.new_tensor(0.0)
        min_scale = scaling.min(dim=1).values  # (N,) 每点最短轴
        return min_scale.mean()

    def get_loss_normal(self, depth_mean, viewpoint):
        prior_normal = viewpoint.normal.cuda()
        prior_normal = prior_normal.reshape(3, *depth_mean.shape[-2:]).permute(
            1, 2, 0
        )
        prior_normal_normalized = torch.nn.functional.normalize(
            prior_normal, dim=-1
        )

        normal_mean, _ = self.depth_to_normal(
            viewpoint, depth_mean, world_frame=False
        )
        normal_error = 1 - (prior_normal_normalized * normal_mean).sum(dim=-1)
        normal_error[prior_normal.norm(dim=-1) < 0.2] = 0
        return normal_error.mean()

    def depth_to_normal(self, view, depth, world_frame=False):
        """
        view: view camera
        depth: depthmap
        """

        points = self.depths_to_points(view, depth, world_frame).reshape(
            *depth.shape[1:], 3
        )
        normal_map = torch.zeros_like(points)
        dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
        dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
        normal_map[1:-1, 1:-1, :] = torch.nn.functional.normalize(
            torch.cross(dx, dy, dim=-1), dim=-1
        )

        return normal_map, points

    def get_loss_flattening(
        self, normal, viewpoint, opacity=None, opacity_threshold=0.95
    ):
        """边缘感知单视图法向损失（参考 PGSR，并考虑不透明度累计）。

        使用 viewpoint.normal（由真实深度 PGSR 方式计算）作为先验。
        损失采用 1 - cos(N_prior, N_render)，再乘边缘权重 (1-grad)^2。
        normal 来自渲染器（compute_shortest_axis_view 的软加权），梯度会反传到高斯的 rotation 与 scale，
        因此该损失会驱动高斯朝向与先验法向一致并压扁最短轴。
        opacity_threshold：仅在不透明度 > 该值的像素上计算。
        """
        rendered_n = torch.nn.functional.normalize(
            normal.permute(1, 2, 0), dim=-1
        )  # (H, W, 3)
        H, W = rendered_n.shape[:2]

        prior_n = viewpoint.normal
        if isinstance(prior_n, np.ndarray):
            prior_n = torch.from_numpy(prior_n)
        prior_n = prior_n.to(device=rendered_n.device, dtype=rendered_n.dtype)

        # 支持 CHW / HWC 两种格式，统一到 CHW
        if prior_n.ndim == 3 and prior_n.shape[0] == 3:
            prior_chw = prior_n
        elif prior_n.ndim == 3 and prior_n.shape[-1] == 3:
            prior_chw = prior_n.permute(2, 0, 1)
        else:
            return rendered_n.new_tensor(0.0)

        if prior_chw.shape[1] != H or prior_chw.shape[2] != W:
            prior_chw = torch.nn.functional.interpolate(
                prior_chw.unsqueeze(0), size=(H, W), mode="nearest"
            ).squeeze(0)
        prior_n = torch.nn.functional.normalize(prior_chw.permute(1, 2, 0), dim=-1)

        # 图像梯度权重 w = (1 - grad_norm)^2
        image = viewpoint.original_image.to(
            device=rendered_n.device, dtype=rendered_n.dtype
        )
        gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        gx = torch.zeros_like(gray)
        gy = torch.zeros_like(gray)
        gx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
        gy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
        grad = torch.sqrt(gx * gx + gy * gy + 1e-8)
        grad_norm = grad / (grad.max().detach() + 1e-8)
        weight = (1.0 - grad_norm).pow(2)

        # 有效像素掩码：两路法向都有效、先验非零，且（若提供 opacity）累计不透明度足够大
        prior_norm = torch.linalg.norm(prior_n, dim=-1)
        valid = (
            torch.isfinite(rendered_n).all(dim=-1)
            & torch.isfinite(prior_n).all(dim=-1)
            & (prior_norm > 1e-3)
        )
        if opacity is not None:
            op = opacity.squeeze()
            if op.shape != valid.shape:
                op = torch.nn.functional.interpolate(
                    op.unsqueeze(0).unsqueeze(0), size=(H, W), mode="nearest"
                ).squeeze()
            valid = valid & (op.to(valid.device) > opacity_threshold)

        # 余弦损失 1 - cos(N_prior, N_render)，对方向更敏感、梯度更稳定，利于法向快速收敛
        cos_sim = (rendered_n * prior_n).sum(dim=-1).clamp(-1.0, 1.0)
        loss_per_pixel = 1.0 - cos_sim
        n_valid = valid.sum().item()
        if n_valid > 0:
            loss = (weight * loss_per_pixel)[valid].mean()
        else:
            loss = (0.0 * rendered_n.sum()).squeeze()
        return loss

    def depths_to_points(self, view, depthmap, world_frame):
        import math

        W, H = view.image_width, view.image_height
        fx = W / (2 * math.tan(view.FoVx / 2.0))
        fy = H / (2 * math.tan(view.FoVy / 2.0))
        intrins = (
            torch.tensor([[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]])
            .float()
            .cuda()
        )
        grid_x, grid_y = torch.meshgrid(
            torch.arange(W, device="cuda").float() + 0.5,
            torch.arange(H, device="cuda").float() + 0.5,
            indexing="xy",
        )
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(
            -1, 3
        )
        if world_frame:
            c2w = (view.world_view_transform.T).inverse()
            rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
            rays_o = c2w[:3, 3]
            points = depthmap.reshape(-1, 1) * rays_d + rays_o
        else:
            rays_d = points @ intrins.inverse().T
            points = depthmap.reshape(-1, 1) * rays_d
        return points

    def initialize(self, cur_frame_idx, viewpoint):

        self.initialized = True
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        self.mapped_video_idxs = []
        self.mapped_kf_idxs = []

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

    def add_new_keyframe(self, cur_frame_idx, idx, depth=None, opacity=None):
        """步骤：记录 mapped 索引→取有效 RGB 区域→单目时根据 depth/opacity 得 initial_depth，否则用 viewpoint.depth；动态时对非 motion 置 0。"""
        rgb_boundary_threshold = self.config["mapping"]["Training"][
            "rgb_boundary_threshold"
        ]
        self.mapped_video_idxs.append(cur_frame_idx)
        self.mapped_kf_idxs.append(idx)
        # monodepth=load_mono_depth(idx, self.save_dir).to(self.device)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        if getattr(self, "save_debug_visualization", False):
            with torch.no_grad():
                output_dir = os.path.join("output", "sitting2_xyz_image")
                os.makedirs(output_dir, exist_ok=True)
                depth1 = gt_img.permute(1, 2, 0).cpu().detach().numpy()
                depth_normalized = (
                    (depth1 - depth1.min()) / (depth1.max() - depth1.min()) * 255
                )
                depth_normalized = depth_normalized.astype(np.uint8)
                cv2.imwrite(
                    os.path.join(output_dir, f"image_test_{idx:04d}.png"),
                    depth_normalized,
                )
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels

            return initial_depth.cpu().numpy()[0]

        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)

        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels

        if self.dynamic_model:
            initial_depth = (
                initial_depth.detach().clone()
            )  # change 0 region according to opacity rendering
            initial_depth[0][~viewpoint.motion_mask.cpu().numpy()] = 0

        return initial_depth[0].cpu().numpy()

    def _align_visibility_mask(self, visibility_mask, target_numel, device=None):
        """Align cached visibility mask length to current gaussian count."""
        vis = visibility_mask.reshape(-1)
        if vis.dtype != torch.bool:
            vis = vis > 0
        if device is not None and vis.device != device:
            vis = vis.to(device)
        if vis.numel() == target_numel:
            return vis
        if vis.numel() > target_numel:
            return vis[:target_numel]
        pad = torch.zeros(target_numel - vis.numel(), dtype=torch.bool, device=vis.device)
        return torch.cat((vis, pad), dim=0)

    def _get_aligned_visibility_pair(self, cur_visibility, ref_visibility):
        target_numel = cur_visibility.numel()
        cur_vis = self._align_visibility_mask(cur_visibility, target_numel)
        ref_vis = self._align_visibility_mask(
            ref_visibility,
            target_numel,
            device=cur_vis.device,
        )
        return cur_vis, ref_vis

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        """步骤：根据平移与重叠比例判定是否为新关键帧。"""
        kf_translation = self.config["mapping"]["Training"]["kf_translation"]
        kf_min_translation = self.config["mapping"]["Training"]["kf_min_translation"]
        kf_overlap = self.config["mapping"]["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        # multiply by median depth in rgb-only setting to account for scale ambiguity
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        last_vis = occ_aware_visibility.get(last_keyframe_idx, None)
        if last_vis is None:
            last_vis = torch.zeros_like(cur_frame_visibility_filter)
        cur_vis, last_vis = self._get_aligned_visibility_pair(
            cur_frame_visibility_filter, last_vis
        )
        union = torch.logical_or(cur_vis, last_vis).count_nonzero()
        intersection = torch.logical_and(cur_vis, last_vis).count_nonzero()
        point_ratio_2 = (
            intersection.float() / union.float() if union.item() > 0 else 0.0
        )
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        """步骤：当前帧入窗→按重叠剔除→超窗大则按 inv_dist 剔除一帧，返回新 window 与被移除帧。"""
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            if kf_idx not in occ_aware_visibility:
                continue
            # szymkiewicz–simpson coefficient
            cur_vis, kf_vis = self._get_aligned_visibility_pair(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            )
            intersection = torch.logical_and(cur_vis, kf_vis).count_nonzero()
            denom = min(
                cur_vis.count_nonzero(),
                kf_vis.count_nonzero(),
            )
            point_ratio_2 = (
                intersection.float() / denom.float() if denom.item() > 0 else 0.0
            )
            cut_off = (
                self.config["mapping"]["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["mapping"]["Training"]
                else 0.4
            )
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.window_size:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    # --------------- 9. 主循环与收尾 ---------------
    def run(self, stream: BaseDataset):
        """
        Trigger mapping process, get estimated pose and depth from tracking process,
        send continue signal to tracking process when the mapping of the current frame finishes.
        """
        config = self.config
        self.stream = stream
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.frame_reader.fx,
            fy=self.frame_reader.fy,
            cx=self.frame_reader.cx,
            cy=self.frame_reader.cy,
            W=self.frame_reader.W_out,
            H=self.frame_reader.H_out,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)

        num_frames = len(self.frame_reader)

        # Initialize list to keep track of Keyframes
        self.keyframe_idxs = []  #
        self.video_idxs = []  # keyframe numbering (note first
        # keyframe for mapping is the 7th keyframe in total)
        self.is_kf = dict()  # keys are video_idx and value is boolean. This prevents trying to deform frames that were never mapped.
        # this is only a problem when the last keyframe is not mapped as this would otherwise be handled by the code.

        # Init Variables to keep track of ground truth poses and runtimes
        self.gt_w2c_all_frames = []

        init = True
        # print("init")
        # Define first frame pose
        _, color, _, first_frame_c2w, _, normal, mono, static_msk = self.frame_reader[0]
        intrinsics = as_intrinsics_matrix(self.frame_reader.get_intrinsic()).to(
            self.device
        )

        # Create dictionary which stores the depth maps from the previous iteration
        # This depth is used during map deformation if we have missing pixels
        self.depth_dict = dict()
        # global camera dictionary - updated during mapping.
        self.cameras = dict()
        self.depth_dict = dict()

        while 1:
            frame_info = self.pipe.recv()
            idx = frame_info["timestamp"]  # frame index
            video_idx = frame_info["video_idx"]  # keyframe index
            is_finished = frame_info["end"]

            if is_finished:
                print("Done with Mapping and Tracking")
                break

            if self.verbose:
                print(Fore.GREEN)
                print("Mapping Frame ", idx)
                print(Style.RESET_ALL)

            self.keyframe_idxs.append(idx)
            self.video_idxs.append(video_idx)

            _, color, depth_gt, c2w_gt, motion_mask, normal, mono, static_msk = (
                self.frame_reader[idx]
            )
            mono_depth = load_mono_depth(idx, self.save_dir).to(self.device)
            color, c2w_gt = color.to(self.device), c2w_gt.to(self.device)
            depth_gt_numpy = depth_gt.numpy()
            depth, w2c, invalid = self.get_w2c_and_depth(
                video_idx,
                idx,
                mono_depth,
                motion_mask,
                depth_gt_numpy,
                normal,
                mono,
                static_msk,
                init=False,
            )

            w2c_gt = torch.linalg.inv(c2w_gt)
            if invalid:
                print("WARNING: Too few valid pixels from droid depth")
                # online glorieslam pose and depth
                data = {
                    "gt_color": color.squeeze(),
                    "glorie_depth": depth.cpu().numpy(),
                    "glorie_pose": w2c,
                    "gt_pose": w2c_gt,
                    "idx": video_idx,
                }
                self.is_kf[video_idx] = False
                viewpoint = Camera.init_from_dataset(
                    self.frame_reader, video_idx, idx, data, projection_matrix
                )
                # update the estimated pose to be the glorie pose
                viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
                viewpoint.compute_grad_mask(self.config)
                # Dictionary of Camera objects at the frame index
                # self.cameras contains all cameras.
                self.cameras[video_idx] = viewpoint
                self.pipe.send("continue")
                continue
            self.gt_w2c_all_frames.append(w2c_gt)
            # 步骤3：建当前帧相机与视角，写 cameras[video_idx]
            data = {
                "gt_color": color.squeeze(),
                "glorie_depth": depth.cpu().numpy(),
                "glorie_pose": w2c,
                "gt_pose": w2c_gt,
                "idx": video_idx,
            }
            viewpoint = Camera.init_from_dataset(
                self.frame_reader, video_idx, idx, data, projection_matrix
            )
            viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
            viewpoint.compute_grad_mask(self.config)
            self.cameras[video_idx] = viewpoint
            if self.dynamic_model:
                self.gaussians.deform.deform.reg_loss = 0.0
            # 步骤4：首帧则初始化地图、加关键帧、initialize_map、initialize_network 后 continue
            if init:
                self.initialize(video_idx, viewpoint)

                self.printer.print("Resetting the system", FontColor.MAPPER)
                self.reset()
                self.current_window.append(video_idx)
                # Add first depth map to depth dictionary - important for the first deformation
                # of the first frame
                self.depth_dict[video_idx] = depth
                self.is_kf[video_idx] = True  # we map the first keyframe (after warmup)
                self.viewpoints[video_idx] = viewpoint
                depth = self.add_new_keyframe(video_idx, idx)
                self.add_next_kf(
                    video_idx,
                    idx,
                    viewpoint,
                    depth_map=depth,
                    init=True,
                )
                self.initialize_map(video_idx, idx, viewpoint)
                # if self.dynamic_model and self.dystart == 11:
                #     print("dynamic0")
                # if self.dynamic_model and self.dystart ==video_idx:
                self.initialize_network(video_idx, viewpoint)
                init = False
                self.pipe.send("continue")
                continue
            # 步骤5：非首帧——渲染取中值深度，做关键帧判定与窗口更新
            dxyz, d_rot, d_scale = 0, 0, 0
            render_pkg = render(
                viewpoint,
                self.gaussians,
                self.pipeline_params,
                self.background,
                dynamic=False,
                dx=dxyz,
                ds=d_scale,
                dr=d_rot,
            )
            self.median_depth = get_median_depth(
                render_pkg["depth"], render_pkg["opacity"]
            )
            last_keyframe_idx = self.current_window[0]
            if len(self.keyframe_idxs) >= 4:
                last_idx = self.keyframe_idxs[-2]
            else:
                last_idx = 0
            curr_visibility = (render_pkg["n_touched"] > 0).long()

            create_kf = (
                self.is_keyframe(
                    video_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                or (idx - last_idx) >= 2
            )
            if len(self.current_window) < self.window_size:
                # When we have not filled up the keyframe window size
                # we rely on just the covisibility thresholding, not the
                # translation thresholds.
                last_vis = self.occ_aware_visibility.get(last_keyframe_idx, None)
                if last_vis is None:
                    last_vis = torch.zeros_like(curr_visibility)
                curr_vis, last_vis = self._get_aligned_visibility_pair(
                    curr_visibility, last_vis
                )
                union = torch.logical_or(curr_vis, last_vis).count_nonzero()
                intersection = torch.logical_and(curr_vis, last_vis).count_nonzero()
                point_ratio = (
                    intersection.float() / union.float() if union.item() > 0 else 0.0
                )

                create_kf = (
                    point_ratio < self.config["mapping"]["Training"]["kf_overlap"]
                    or (idx - last_keyframe_idx) >= 2
                )

            if create_kf:
                self.current_window, removed = self.add_to_window(
                    video_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                    self.current_window,
                )
                self.is_kf[video_idx] = True
            else:
                self.is_kf[video_idx] = False
                self.pipe.send("continue")
                continue
            last_idx = self.keyframe_idxs[-1]
            # 步骤6：遍历已有关键帧，更新位姿与深度（含 depth_dict），非最新帧则 update_mapping_points
            for keyframe_idx, frame_idx in zip(self.video_idxs, self.keyframe_idxs):
                # need to update depth_dict even if the last idx since this is important
                # for the first deformation of the keyframe
                _, _, depth_gtd, _, motion_mask, normal, mono, static_msk = (
                    self.frame_reader[frame_idx]
                )
                depth_gt_numpy = depth_gtd.cpu().numpy()
                mono_depth = load_mono_depth(frame_idx, self.save_dir).to(self.device)
                depth_gtd = depth_gtd.to(mono_depth.device)

                depth_temp, w2c_temp, invalid = self.get_w2c_and_depth(
                    keyframe_idx,
                    frame_idx,
                    mono_depth,
                    motion_mask,
                    depth_gt_numpy,
                    normal,
                    mono,
                    static_msk,
                    init=False,
                )
                if keyframe_idx not in self.depth_dict and self.is_kf[keyframe_idx]:
                    self.depth_dict[keyframe_idx] = depth_temp

                # No need to move the latest pose and depth
                if frame_idx != last_idx:
                    # Update tracking parameters
                    w2c_old = torch.cat(
                        (
                            self.cameras[keyframe_idx].R,
                            self.cameras[keyframe_idx].T.unsqueeze(-1),
                        ),
                        dim=1,
                    )
                    w2c_old = torch.cat(
                        (w2c_old, torch.tensor([[0, 0, 0, 1]], device="cuda")), dim=0
                    )
                    self.cameras[keyframe_idx].update_RT(
                        w2c_temp[:3, :3], w2c_temp[:3, 3]
                    )
                    # Update depth for viewpoint
                    self.cameras[keyframe_idx].depth = depth_temp.cpu().numpy()

                    if keyframe_idx in self.viewpoints:
                        # Update tracking parameters
                        self.viewpoints[keyframe_idx].update_RT(
                            w2c_temp[:3, :3], w2c_temp[:3, 3]
                        )
                        # Update depth for viewpoint
                        self.viewpoints[keyframe_idx].depth = depth_temp.cpu().numpy()

                    # Update mapping parameters
                    if self.move_points and self.is_kf[keyframe_idx]:
                        if invalid:
                            # if the frame was invalid, we don't update the depth old and just do a rigid correction for this frame
                            self.update_mapping_points(
                                keyframe_idx,
                                w2c_temp,
                                w2c_old,
                                depth_temp,
                                self.depth_dict[keyframe_idx],
                                intrinsics,
                                method="rigid",
                            )
                        else:
                            self.update_mapping_points(
                                keyframe_idx,
                                w2c_temp,
                                w2c_old,
                                depth_temp,
                                self.depth_dict[keyframe_idx],
                                intrinsics,
                            )
                            self.depth_dict[keyframe_idx] = (
                                depth_temp  # line does not matter since it is the last deformation anyway
                            )
            # 步骤7：加入当前帧到 viewpoints、add_new_keyframe、add_next_kf；可选更新动态掩码
            self.viewpoints[video_idx] = viewpoint
            depth = self.add_new_keyframe(video_idx, idx)
            self.add_next_kf(video_idx, idx, viewpoint, depth_map=depth, init=False)
            # 动态掩码已移至前端（tracker）计算并保存为图片；后端通过 dataset 读盘获得，此处不再调用

            self.initialized = self.initialized or (
                len(self.current_window) == self.window_size
            )
            # 步骤8：构建 keyframe_optimizers，执行 map(prune=False) 与 map(prune=True)，清理并发送 continue
            opt_params = []
            frames_to_optimize = self.config["mapping"]["Training"]["pose_window"]
            if self.verbose:
                print("current_window", len(self.current_window))
            for cam_idx in range(len(self.current_window)):
                if self.current_window[cam_idx] == 0:
                    # Do not add GT frame pose for optimization
                    continue
                viewpoint = self.viewpoints[self.current_window[cam_idx]]
                # if not self.gt_camera and self.config["mapping"]["BA"]:
                if cam_idx < frames_to_optimize:
                    opt_params.append(
                        {
                            "params": [viewpoint.cam_rot_delta],
                            "lr": self.config["mapping"]["Training"]["lr"][
                                "cam_rot_delta"
                            ]
                            * 0.5,
                            "name": "rot_{}".format(viewpoint.uid),
                        }
                    )
                    opt_params.append(
                        {
                            "params": [viewpoint.cam_trans_delta],
                            "lr": self.config["mapping"]["Training"]["lr"][
                                "cam_trans_delta"
                            ]
                            * 0.5,
                            "name": "trans_{}".format(viewpoint.uid),
                        }
                    )

                opt_params.append(
                    {
                        "params": [viewpoint.exposure_a],
                        "lr": 0.01,
                        "name": "exposure_a_{}".format(viewpoint.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint.exposure_b],
                        "lr": 0.01,
                        "name": "exposure_b_{}".format(viewpoint.uid),
                    }
                )
            self.keyframe_optimizers = torch.optim.Adam(opt_params)

            dynamic_enabled = self.config["mapping"]["model_params"].get(
                "dynamic_model", False
            )

            map_iters = self.config["mapping"]["Training"].get("map_iters_per_kf", 400)
            self.map(
                stream=stream,
                idx1=idx,
                current_window=self.current_window,
                iters=map_iters,
                dynamic_network=dynamic_enabled,
                prune=False,
            )
            # self.map_static(current_window=self.current_window, iters=50,idx1=idx)  # 静态地图优化
            self.map(
                stream=stream,
                idx1=idx,
                current_window=self.current_window,
                prune=True,
                dynamic_network=dynamic_enabled,
            )
            del render_pkg, depth, w2c, depth_temp, w2c_temp
            torch.cuda.empty_cache()
            self.cleanup(video_idx)
            self.pipe.send("continue")

    def cleanup(self, cur_frame_idx):
        if cur_frame_idx % 1 == 0:
            torch.cuda.empty_cache()

    def initialize_network(self, cur_frame_idx, viewpoint, update_gaussians=False):
        """与 Splat-slam-try-copy 完全一致：expand_time + deform.step + render(仅 dx,ds,dr)，100 次迭代。"""
        from src.utils.Printer import get_msg_prefix

        prefix = get_msg_prefix(FontColor.MAPPER)
        if cur_frame_idx == self.dystart:
            inited = self.gaussians.create_node_from_depth(
                viewpoint, self.opt_params, self.sc_params
            )
            if not inited:
                return
        pbar = tqdm(
            range(100),
            desc=prefix + "Init deform network",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            position=0,
            leave=True,
        )
        for mapping_iteration in pbar:
            dxyz, d_rot, d_scale, _, _ = self._get_deform_d_values(
                viewpoint, use_noise=False, detach_outputs=False, iteration=0
            )
            render_pkg = self._render_viewpoint(viewpoint, dxyz, d_scale, d_rot)
            image, _, _, _, depth, opacity, _, _ = self._unpack_render_pkg(render_pkg)
            loss_init = get_loss_mapping(
                self.config["mapping"],
                image,
                depth,
                viewpoint,
                opacity,
                initialization=True,
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.deform.optimizer.step()
                self.gaussians.deform.optimizer.zero_grad(set_to_none=True)
                if update_gaussians:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
        vis_render_process(
            self.gaussians,
            self.pipeline_params,
            self.background,
            viewpoint,
            viewpoint.uid,
            self.save_dir,
            out_dir="init_network",
            mask=None,
            dynamic=True,
            save_debug=getattr(self, "save_debug_visualization", False),
        )
