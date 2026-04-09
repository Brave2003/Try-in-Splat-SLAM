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
#from encodings.punycode import selective_find
from PIL import Image
import cv2
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import torch
import random
import torch.nn as nn
#from jsonschema.benchmarks.const_vs_enum import invalid
from pygments.lexer import combined
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
from thirdparty.gaussian_splatting.gaussian_renderer import render,render_flow
from thirdparty.gaussian_splatting.utils.general_utils import rotation_matrix_to_quaternion, quaternion_multiply
from thirdparty.gaussian_splatting.utils.loss_utils import l1_loss, ssim
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from thirdparty.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from thirdparty.lietorch.examples.rgbdslam.rgbd_benchmark.evaluate_rpe import find_closest_index
from thirdparty.monogs.utils.pose_utils import update_pose
from thirdparty.monogs.utils.slam_utils import get_loss_mapping, get_median_depth,get_loss_tracking,depth_loss_dpt
from thirdparty.monogs.utils.camera_utils import Camera
#from utils.eval_utils import  save_gaussians
from argparse import ArgumentParser
from arguments import ModelHiddenParams
def vis_render_process(gaussians, pipeline_params, background, viewpoint, cur_frame_idx, save_dir, out_dir="map",
                    mask=None, dynamic=False):
    with torch.no_grad():
        if dynamic:
            time_input = gaussians.deform.deform.expand_time(viewpoint.fid)
            d_values = gaussians.deform.step(gaussians.get_dygs_xyz.detach(), time_input,
                                            iteration=0, feature=None,
                                            motion_mask=gaussians.motion_mask,
                                            camera_center=viewpoint.camera_center,
                                            time_interval=gaussians.time_interval)
            dxyz = d_values['d_xyz']
            d_rot, d_scale = d_values['d_rotation'], d_values['d_scaling']
            d_opac = d_values.get('d_opacity')
            d_color = d_values.get('d_color')
            # print("scale: ", d_scale)
        else:
            dxyz, d_rot, d_scale = 0, None, 0
            d_opac, d_color = None, None
        render_pkg = render(
            viewpoint,
            gaussians,
            pipeline_params,
            background,
            dynamic=False,
            dx=dxyz,
            ds=d_scale,
            dr=d_rot,
            do=d_opac,
            dc=d_color,
        )

        viz_im = torch.clip(render_pkg["render"].permute(1, 2, 0).detach().cpu(), 0, 1)


        h, w, _ = viz_im.shape
        fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
        cax = ax.imshow(viz_im)
        ax.axis('off')

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        os.makedirs(save_dir, exist_ok=True)
        process_dir = os.path.join(save_dir, out_dir)
        os.makedirs(process_dir, exist_ok=True)
        save_path = os.path.join(process_dir, f"{cur_frame_idx}.png")
        viz_im_np= np.array(viz_im)
        cv2.imwrite(save_path, viz_im_np)
        plt.savefig(save_path)
        plt.close()
        return
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

    # ---------------------------------------------------------------------
    # Section A: Lifecycle and Global Hyper-Parameters
    # ---------------------------------------------------------------------

    def __init__(self, slam, pipe: Connection):
        # setup seed
        setup_seed(slam.cfg["setup_seed"])
        torch.autograd.set_detect_anomaly(True)

        self.config = slam.cfg
        self.printer: Printer = slam.printer
        if self.config['only_tracking']:
            return

        self.pipe = pipe
        self.verbose = slam.verbose

        # Runtime handles and containers
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
        self._last_dynamic_gaussians = None

        # Scene/tracking state
        training_cfg = self.config["mapping"]["Training"]
        self.dystart = training_cfg["dystart"] if "dystart" in training_cfg else 11
        self.video: DepthVideo = slam.video
        self.monocular = not self.initialized

        # Config-derived params
        mapping_cfg = self.config["mapping"]
        model_cfg = mapping_cfg["model_params"]
        self.use_spherical_harmonics = training_cfg["spherical_harmonics"]
        self.dynamic_model = model_cfg["dynamic_model"]

        # Optimizer/model argument packs
        model_params = munchify(self.config["mapping"]["model_params"])
        opt_params = munchify(self.config["mapping"]["opt_params"])
        pipeline_params = munchify(self.config["mapping"]["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )
        self.use_normal = mapping_cfg.get("use_normal", False)

        parser = ArgumentParser(description="Training script parameters")
        hp = ModelHiddenParams(parser)
        hp = merge_hparams(hp, self.config["mapping"])
        self.sc_params = hp

        # Gaussian model initialization
        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0
        self.gaussians = GaussianModel(
            model_params.sh_degree,
            config=self.config,
            args=hp,
            init_deform=self.dynamic_model,
        )
        self.gaussians.init_lr(6.0)

        # Scale-shift and helper caches
        self.st_predicted = {}
        self.list = []
        self.first_d = []
        self.new_scale_alignFrame0 = dict()
        self._image_coord_cache = {}
        self._st_compare_counter = 0

        static_msk = np.ones((384, 512), dtype=bool)
        self.gaussians.training_setup(opt_params)

        # Global rendering/mapping state
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.longer = mapping_cfg["long"]
        self.cameras_extent = 6.0
        self.dyratio = 0
        self.shift = mapping_cfg["shift"]
        self.s = mapping_cfg["s"]

        # Hyper-parameters depending on camera extent and mapping config
        self.set_hyperparams()

        self.device = torch.device(self.config['device'])
        self.static_msk = torch.from_numpy(static_msk).to(self.device)
        self.frame_reader = get_dataset(self.config, device=self.device)

        if self.dynamic_model:
            self.gaussians.deform.train_setting(hp)
            self.gaussians.time_interval = 1 / len(self.frame_reader)

    def set_pipe(self, pipe):
        self.pipe = pipe

    def set_hyperparams(self):
        mapping_config = self.config["mapping"]
        training_config = mapping_config["Training"]

        # Core mapping schedule
        self.gt_camera = training_config["gt_camera"]
        self.init_itr_num = training_config["init_itr_num"]
        self.init_gaussian_update = training_config["init_gaussian_update"]
        self.init_gaussian_reset = training_config["init_gaussian_reset"]
        self.init_gaussian_th = training_config["init_gaussian_th"]
        self.dynamic_init_gaussian_th = float(
            training_config.get("dynamic_init_gaussian_th", self.init_gaussian_th)
        )
        self.init_gaussian_extent = (
            self.cameras_extent * training_config["init_gaussian_extent"]
        )
        self.mapping_itr_num = training_config["mapping_itr_num"]

        # Gaussian maintenance schedule
        self.gaussian_update_every = training_config["gaussian_update_every"]
        self.gaussian_update_offset = training_config["gaussian_update_offset"]
        self.gaussian_th = training_config["gaussian_th"]
        self.dynamic_gaussian_th = float(
            training_config.get("dynamic_gaussian_th", self.gaussian_th)
        )
        self.gaussian_extent = (
            self.cameras_extent * training_config["gaussian_extent"]
        )
        self.gaussian_reset = training_config["gaussian_reset"]
        self.size_threshold = training_config["size_threshold"]
        self.window_size = training_config["window_size"]
        self.prune_coviz = int(training_config.get("prune_coviz", 3))
        self.dynamic_prune_coviz = int(training_config.get("dynamic_prune_coviz", 1))
        self.dynamic_prune_grace = int(
            training_config.get("dynamic_prune_grace", 2 * self.window_size)
        )
        self.keep_dynamic_gaussians = bool(
            training_config.get("keep_dynamic_gaussians", True)
        )
        self.protect_dynamic_out_of_view = bool(
            training_config.get("protect_dynamic_out_of_view", True)
        )
        self.preserve_invisible_gaussians = bool(
            training_config.get("preserve_invisible_gaussians", True)
        )
        self.freeze_dynamic_when_invisible = bool(
            training_config.get("freeze_dynamic_when_invisible", True)
        )

        # Loss weights
        self.depth_order_loss_weight = training_config.get("depth_order_loss_weight", 0.1)
        self.normal_loss_weight = training_config.get("normal_loss_weight", 0.01)
        self.dynamic_phase_ratio = float(training_config.get("dynamic_phase_ratio", 0.5))
        self.dynamic_transition_ratio = float(training_config.get("dynamic_transition_ratio", 0.2))
        self.flow_loss_start = float(training_config.get("flow_loss", 0.0))
        self.flow_loss_end = float(training_config.get("flow_loss_fine", self.flow_loss_start))

        self.save_dir = self.config['data']['output'] + '/' + self.config['scene']

        self.move_points = self.config['mapping']['move_points']
        self.online_plotting = self.config['mapping']['online_plotting']
        self.node_traj_vis = mapping_config.get("node_traj_vis", False)
        self.node_traj_vis_every = int(mapping_config.get("node_traj_vis_every", 1))
        self.node_traj_t_samples = int(
            mapping_config.get("dynamic_traj_t_samples", mapping_config.get("node_traj_t_samples", 12))
        )
        self.node_traj_max_nodes = int(
            mapping_config.get("dynamic_traj_max_points", mapping_config.get("node_traj_max_nodes", 96))
        )
        self.node_traj_final_vis = bool(mapping_config.get("node_traj_final_vis", True))
        self.node_traj_final_t_samples = int(
            mapping_config.get("node_traj_final_t_samples", 0)
        )
        self.node_traj_video_fps = int(mapping_config.get("node_traj_video_fps", 12))
        self.node_traj_video_tail = int(mapping_config.get("node_traj_video_tail", 16))
        self.node_traj_gif = bool(mapping_config.get("node_traj_gif", True))
        self.node_traj_gif_fps = int(
            mapping_config.get("node_traj_gif_fps", mapping_config.get("node_traj_video_fps", 12))
        )
        self.node_traj_gif_max_frames = int(mapping_config.get("node_traj_gif_max_frames", 0))
        self.node_traj_subset_seed = int(mapping_config.get("node_traj_subset_seed", -1))
        self.node_traj_3d_margin = float(mapping_config.get("node_traj_3d_margin", 0.18))
        self.node_traj_3d_elev = float(mapping_config.get("node_traj_3d_elev", 22.0))
        self.node_traj_3d_azim_start = float(mapping_config.get("node_traj_3d_azim_start", -58.0))
        self.node_traj_3d_azim_spin = float(mapping_config.get("node_traj_3d_azim_spin", 48.0))
        self.st_flat_keep_percent = float(mapping_config.get("st_flat_keep_percent", 80.0))

    def add_next_kf(self, frame_idx,idx, viewpoint, init=False, scale=2.0, depth_map=None):
        # This function computes the new Gaussians to be added given a new keyframe
        #print("depth",depth_map)
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map,
        )
        if frame_idx == self.dystart:
            self.gaussians.extend_from_pcd_seq(
                viewpoint, kf_id=frame_idx, init=True, scale=scale, depthmap=depth_map, add_dygs=True
            )

    # ---------------------------------------------------------------------
    # Section B: Keyframe and Geometry State Update
    # ---------------------------------------------------------------------

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = True
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)

    def update_mapping_points(self, frame_idx, w2c, w2c_old, depth, depth_old, intrinsics, method=None):
        if method == "rigid":
            # just move the points according to their SE(3) transformation without updating depth
            frame_idxs = self.gaussians.unique_kfIDs  # idx which anchored the set of points
            frame_mask = (frame_idxs == frame_idx)  # global variable
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
            self.gaussians._xyz = self.gaussians.replace_tensor_to_optimizer(means, "xyz")["xyz"]
            # transform the corresponding rotation matrices
            rots = self.gaussians.get_rotation.detach()
            # Convert transformation to quaternion
            transformation = rotation_matrix_to_quaternion(transformation.unsqueeze(0))
            rots[frame_mask] = quaternion_multiply(transformation.expand_as(rots[frame_mask]), rots[frame_mask])

            with torch.no_grad():
                self.gaussians._rotation = self.gaussians.replace_tensor_to_optimizer(rots, "rotation")["rotation"]
        else:
            # Update pose and depth by projecting points into the pixel space to find updated correspondences.
            # This strategy also adjusts the scale of the gaussians to account for the distance change from the camera
            depth = depth.to(self.device)
            while depth.dim() > 2:
                if depth.shape[0] == 1:
                    depth = depth.squeeze(0)
                else:
                    self.printer.print(
                        f"Skip update_mapping_points[{frame_idx}] due to invalid depth shape: {tuple(depth.shape)}",
                        FontColor.MAPPER,
                    )
                    return
            frame_idxs = self.gaussians.unique_kfIDs  # idx which anchored the set of points
            frame_mask = (frame_idxs == frame_idx)  # global variable
            if frame_mask.sum() == 0:
                return
            if depth.dim() != 2:
                self.printer.print(
                    f"Skip update_mapping_points[{frame_idx}] due to invalid depth shape: {tuple(depth.shape)}",
                    FontColor.MAPPER,
                )
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
            pixel_locations[:, 0] = torch.clamp(pixel_locations[:, 0], min=0, max=width - 1)
            pixel_locations[:, 1] = torch.clamp(pixel_locations[:, 1], min=0, max=height - 1)
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
            rescale_scale = (1 + 1 / (means_cam[:, 2]) * (depth - depth_old)).unsqueeze(-1)  # shift
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
            self.gaussians._xyz = self.gaussians.replace_tensor_to_optimizer(global_means, "xyz")["xyz"]
            # update the rotation of the gaussians
            rots = self.gaussians.get_rotation.detach()
            # Convert transformation to quaternion
            transformation = rotation_matrix_to_quaternion(transformation.unsqueeze(0))
            rots[frame_mask] = quaternion_multiply(transformation.expand_as(rots[frame_mask]), rots[frame_mask])
            self.gaussians._rotation = self.gaussians.replace_tensor_to_optimizer(rots, "rotation")["rotation"]
            # Update the scale of the Gaussians
            scales = self.gaussians._scaling.detach()
            scales[frame_mask] = scales[frame_mask] + torch.log(rescale_scale)
            self.gaussians._scaling = self.gaussians.replace_tensor_to_optimizer(scales, "scaling")["scaling"]

    # ---------------------------------------------------------------------
    # Section C: Depth/Normal and Scale-Shift Alignment Utilities
    # ---------------------------------------------------------------------

    def init_image_coor(self,height, width):
        cache_key = (height, width, str(self.device))
        if cache_key in self._image_coord_cache:
            return self._image_coord_cache[cache_key]

        x_row = torch.arange(width, dtype=torch.float32, device=self.device)
        y_col = torch.arange(height, dtype=torch.float32, device=self.device)
        x = x_row.repeat(height, 1).unsqueeze(0)
        y = y_col.view(height, 1).repeat(1, width).unsqueeze(0)
        u_u0 = x - width / 2.0
        v_v0 = y - height / 2.0
        self._image_coord_cache[cache_key] = (u_u0, v_v0)
        return u_u0, v_v0
    
    def depth_to_xyz(self,depth, focal_x,focal_y):
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
        start_time = time.perf_counter()
        input_depth = depth_torch.to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        gt_normal = normal_torch.to(self.device, dtype=torch.float32)

        if gt_normal.shape[-2:] != input_depth.shape[-2:]:
            gt_normal = F.interpolate(gt_normal, size=input_depth.shape[-2:], mode="nearest")
        gt_normal = F.normalize(gt_normal, dim=1)

        focal_x = torch.tensor([self_defined_focal_x], dtype=torch.float32, device=self.device)
        focal_y = torch.tensor([self_defined_focal_y], dtype=torch.float32, device=self.device)
        
        s = nn.Parameter(torch.tensor([1.0], dtype=torch.float32, device=self.device))
        t = nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device=self.device))

        optimizer = torch.optim.Adam(
            [
                {"params": s, "lr": 1e-3},
                {"params": t, "lr": 1e-3},
            ]
        )
        last_step_loss = 100000.0
        for step in range(500):
            optimizer.zero_grad(set_to_none=True)
            scaled_depth = s * input_depth + t
            depth_filter = nn.functional.avg_pool2d(scaled_depth, kernel_size=3, stride=1, padding=1)
            depth_filter = nn.functional.avg_pool2d(depth_filter, kernel_size=3, stride=1, padding=1)
            xyz = self.depth_to_xyz(depth_filter, focal_x, focal_y)
            xyz_i = xyz[0, :][None, :, :, :]
            pre_normal = self.get_surface_normalv2(xyz_i).permute((3, 2, 0, 1))
            # 与单目深度先验法向约定对齐（同 before_modify）
            similarity = torch.nn.functional.cosine_similarity(pre_normal, gt_normal, dim=1)
            loss_map = 1.0 - similarity
            loss_normal = torch.nanmean(loss_map)
            loss = loss_normal

            loss.backward()
            optimizer.step()
            if step % 40 == 0:
                if abs(loss.item() - last_step_loss) < 1e-5:
                    break
                last_step_loss = loss.item()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        print(
            "[optimize_st][Optimized] "
            f"s={s.item():.6f}, t={t.item():.6f}, loss={loss.item():.6f}, step={step}, time={elapsed_ms:.2f}ms"
        )
        return s.item(), t.item()

    def get_surface_normalv2(self, xyz, patch_size=3):
        """
        优化后的法向量计算：去除了所有阻碍 backward() 速度的操作
        """
        eps = 1e-8
        b, h, w, c = xyz.shape
        half_patch = patch_size // 2
        
        # 【核心优化 1】：用 F.pad 替代 torch.zeros 和切片赋值。
        # 这是一个零拷贝（Zero-copy）级别的操作，大幅减少显存分配开销和建图时间。
        # Pad 的顺序是从后向前：先 Pad 最后一维 C(不 pad)，再 pad W，再 pad H。
        xyz_pad = F.pad(xyz, (0, 0, half_patch, half_patch, half_patch, half_patch), mode='constant', value=0.0)

        xyz_left = xyz_pad[:, half_patch:half_patch + h, :w, :]  
        xyz_right = xyz_pad[:, half_patch:half_patch + h, -w:, :]  
        xyz_top = xyz_pad[:, :h, half_patch:half_patch + w, :]  
        xyz_bottom = xyz_pad[:, -h:, half_patch:half_patch + w, :]  
        xyz_horizon = xyz_left - xyz_right  
        xyz_vertical = xyz_top - xyz_bottom  

        xyz_left_in = xyz_pad[:, half_patch:half_patch + h, 1:w + 1, :]  
        xyz_right_in = xyz_pad[:, half_patch:half_patch + h, patch_size - 1:patch_size - 1 + w, :]  
        xyz_top_in = xyz_pad[:, 1:h + 1, half_patch:half_patch + w, :]  
        xyz_bottom_in = xyz_pad[:, patch_size - 1:patch_size - 1 + h, half_patch:half_patch + w, :]  
        xyz_horizon_in = xyz_left_in - xyz_right_in  
        xyz_vertical_in = xyz_top_in - xyz_bottom_in  

        n_img_1 = torch.cross(xyz_horizon_in, xyz_vertical_in, dim=3)
        n_img_2 = torch.cross(xyz_horizon, xyz_vertical, dim=3)

        # 【核心优化 2】：彻底消灭布尔掩码 (orient_mask = ... -> tensor[mask] *= -1)
        # 利用数学符号函数 torch.sign() 实现纯张量乘法，绕过低效的掩码求导。
        # 如果内积 > 0，sign=1，乘 -1 翻转；如果内积 < 0，sign=-1，乘 -1 保持原样不变。
        dot_1 = torch.sum(n_img_1 * xyz, dim=3, keepdim=True)
        n_img_1 = n_img_1 * -torch.sign(dot_1 + eps)
        
        dot_2 = torch.sum(n_img_2 * xyz, dim=3, keepdim=True)
        n_img_2 = n_img_2 * -torch.sign(dot_2 + eps)

        n_img1_L2 = torch.sqrt(torch.sum(n_img_1 ** 2, dim=3, keepdim=True) + eps)
        n_img1_norm = n_img_1 / (n_img1_L2 + eps)

        n_img2_L2 = torch.sqrt(torch.sum(n_img_2 ** 2, dim=3, keepdim=True) + eps)
        n_img2_norm = n_img_2 / (n_img2_L2 + eps)

        n_img_aver = n_img1_norm + n_img2_norm
        n_img_aver_L2 = torch.sqrt(torch.sum(n_img_aver ** 2, dim=3, keepdim=True) + eps)
        n_img_aver_norm = n_img_aver / (n_img_aver_L2 + eps)
        
        # 同样干掉最后一处掩码
        dot_aver = torch.sum(n_img_aver_norm * xyz, dim=3, keepdim=True)
        n_img_aver_norm = n_img_aver_norm * -torch.sign(dot_aver + eps)
        
        n_img_aver_norm_out = n_img_aver_norm.permute((1, 2, 3, 0))  

        return n_img_aver_norm_out
        
    def obtain_allimgs_st(self,depth, normal, st_dict, video_idx,self_define_focal_x: float,self_define_focal_y: float):
        depth_torch = depth.detach().to("cpu")
        normal_torch = torch.from_numpy(normal) if isinstance(normal, np.ndarray) else normal.detach().to("cpu")
        video_idx = int(video_idx)

        if video_idx not in st_dict:
            s, t = self.optimize_st(depth_torch, normal_torch, self_define_focal_x, self_define_focal_y)
            st_dict[video_idx] = {"scale": float(s), "shift": float(t)}

        self._update_st_running_mean()

    def _update_st_running_mean(self):
        scales = []
        shifts = []
        for k, v in self.st_predicted.items():
            if not isinstance(k, int) or not isinstance(v, dict):
                continue

            s = v.get("scale", None)
            t = v.get("shift", None)
            if s is None or t is None:
                continue

            s = float(s)
            t = float(t)
            if np.isfinite(s) and np.isfinite(t):
                scales.append(s)
                shifts.append(t)

        if len(scales) > 0:
            self.st_predicted["mean_s"] = float(np.mean(scales))
            self.st_predicted["mean_t"] = float(np.mean(shifts))

    def align_all_frames(self,depth, video_id,st_predicted, new_scale_alignFrame0: dict, static_msk):
        # depth_dirs = glob(os.path.join(base_dir,'GeoWizardOut/depth_npy/*.npy'))
        ref_kf = self.dystart
        if video_id == ref_kf and len(self.list) == 0:
            self.list.append(depth.detach().clone())
        if len(self.list) == 0:
            new_scale_alignFrame0[video_id] = 1.0
            return new_scale_alignFrame0
        reference_depth = self.list[0] 
        reference_depth = reference_depth.to(depth.device)
        # masked_reference_depth = reference_depth
        reference_s = float(st_predicted[ref_kf]["scale"])
        if st_predicted[ref_kf]["shift"] < 0:
            reference_t = -1.1 * float(st_predicted[ref_kf]["shift"])  # 0.9
        else:
            reference_t = float(st_predicted[ref_kf]["shift"])

        scaled_refer_depth_nomsk = reference_s * reference_depth + reference_t
        h, w = scaled_refer_depth_nomsk.shape
        resized_static_msk = F.interpolate(
            static_msk.to(depth.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="nearest",
        ).squeeze(0).squeeze(0) > 0.5

        scaled_refer_depth = scaled_refer_depth_nomsk[resized_static_msk]
        del reference_t,reference_s
        Y = scaled_refer_depth

        depth = depth
        cur_s = float(st_predicted[video_id]["scale"])

        if self.longer:
            cur_t = max(float(st_predicted[video_id]["shift"]),0)
        else:
            if st_predicted[video_id]["shift"] < 0:
                cur_t = -float(st_predicted[video_id]["shift"])
            else:
                cur_t = float(st_predicted[video_id]["shift"])
        if np.isnan(cur_s) or np.isnan(cur_t):
            cur_s = float(st_predicted["mean_s"])
            cur_t = float(st_predicted["mean_t"])

        cur_masked_depth = depth[resized_static_msk]
        scaled_cur_depth = cur_masked_depth * cur_s + 0

        if scaled_cur_depth.numel() == 0 or Y.numel() == 0:
            new_scale_alignFrame0[video_id] = 1.0
        else:
            # Faster and equivalent to single-variable least squares: argmin_a ||a*A - Y||_2^2
            denom = torch.dot(scaled_cur_depth, scaled_cur_depth)
            if torch.abs(denom) < 1e-12:
                solved_scale = 1.0
            else:
                solved_scale = (torch.dot(scaled_cur_depth, Y) / denom).item()

            if solved_scale == 0:
                new_scale_alignFrame0[video_id] = 1.0
            else:
                new_scale_alignFrame0[video_id] = solved_scale

            previous_value = new_scale_alignFrame0.get(ref_kf, None)

            if previous_value is not None and self.longer:

                current_value = new_scale_alignFrame0[video_id]

                if current_value is None or current_value > previous_value*self.s or current_value*self.s<previous_value :
                    new_scale_alignFrame0[video_id] = previous_value
        # print("S,T,align_scale:", cur_s, cur_t, new_scale_alignFrame0[video_id])
        del cur_s,cur_t
        return new_scale_alignFrame0

    def export_scaled_pcd(self,st_predicted, new_scale_alignFrame0, depth, video_id,mean_st=False, ):

        # import open3d as o3d

        if mean_st:
            s = st_predicted["mean_s"]
            t = st_predicted["mean_t"]
        else:
            s = st_predicted[video_id]["scale"]

            if st_predicted[video_id]["shift"] < 0:
                t = -self.shift*st_predicted[video_id]["shift"]#0.6
            else:
                t = st_predicted[video_id]["shift"]
            #t = max(st_predicted[video_id]["shift"], 0)
        depth = depth * s+t
        if video_id in new_scale_alignFrame0:
            depth = depth * new_scale_alignFrame0[video_id]
        return depth


    def get_depth_order_loss(self,render_depth, gt_depth, mask, pair_num=200000, alpha=100, ):
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
        gt_depth=gt_depth.to(render_depth.device)
        # alpha = 100
        gt_depth = gt_depth[mask > 0]  ## N,1
        depthmax = gt_depth.max()
        depthmin = gt_depth.min()
        interval = (depthmax - depthmin) / 10
        # interval = (depthmax-depthmin)/20

        render_depth = render_depth.squeeze(0)[mask > 0]  ## N,1
        index1 = torch.randperm(gt_depth.shape[0])[:pair_num, ]
        index2 = torch.randperm(gt_depth.shape[0])[:pair_num, ]
        index1 = index1.to(render_depth.device)
        index2 = index2.to(render_depth.device)
        threshold_msk = (torch.abs(gt_depth[index1] - gt_depth[index2]) >= interval)
        threshold_msk= threshold_msk.to(render_depth.device)
        index1 = index1[threshold_msk]
        index2 = index2[threshold_msk]

        gt_oder = torch.sign(gt_depth[index1] - gt_depth[index2])
        render_diff = render_depth[index1] - render_depth[index2]

        loss = torch.mean(torch.abs(torch.tanh(alpha * render_diff) - gt_oder))
        return loss

    def get_loss_normal(self,depth_mean, viewpoint):
        prior_normal = viewpoint.normal.cuda()
        prior_normal = prior_normal.reshape(3, *depth_mean.shape[-2:]).permute(1, 2, 0)
        prior_normal_normalized = torch.nn.functional.normalize(prior_normal, dim=-1)

        normal_mean, _ = self.depth_to_normal(viewpoint, depth_mean, world_frame=False)
        normal_error = 1 - (prior_normal_normalized * normal_mean).sum(dim=-1)
        normal_error[prior_normal.norm(dim=-1) < 0.2] = 0
        return normal_error.mean()

    def get_render_normal_loss(self, gt_normal, render_normal):
        """
        计算 Ground Truth 法向量与 3DGS 渲染得到的法向量之间的损失
        """
        gt_normal = gt_normal.squeeze(0)
        if gt_normal is None or render_normal is None:
            return torch.tensor(0.0, device=render_normal.device)

        # 把 gt_normal reshape 成与 render_normal 一致的 [3, H, W]
        if gt_normal.dim() == 1:
            gt_normal = gt_normal.reshape(3, render_normal.shape[1], render_normal.shape[2])
            
        # 很多时候数据集法向可能是 HWC 或是平铺的，稳妥起见确保形状对齐
        if gt_normal.shape != render_normal.shape:
            gt_normal = torch.nn.functional.interpolate(gt_normal.unsqueeze(0), size=render_normal.shape[-2:], mode="nearest").squeeze(0)

        # 归一化 (在通道维度 dim=0 上做 L2 归一化)
        gt_norm = torch.nn.functional.normalize(gt_normal, p=2, dim=0)
        render_norm = torch.nn.functional.normalize(render_normal, p=2, dim=0)

        # 基于余弦相似度的方向损失 (1 - cos(theta)) 
        cos_similarity = (gt_norm * render_norm).sum(dim=0)
        loss_cos = 1.0 - cos_similarity

        l1_loss = torch.abs(gt_norm - render_norm).sum(dim=0)

        # 将两种损失结合 (这里使用 0.8 * cos + 0.2 * L1)
        loss_map = loss_cos + 0.2 * l1_loss

        # 构建 Mask 去除无效区域
        valid_mask = gt_normal.norm(p=2, dim=0) > 0.1
        
        if valid_mask.sum() > 0:
            return loss_map[valid_mask].mean()
        else:
            return loss_map.mean()

    def get_flat_loss(self):
        """
        高斯扁平化损失 (Flatten Loss):
        约束 3DGS 在一个方向（通常是 scale 的最小值）尽可能小，
        使得高斯球变成类似于 '薄片' (splats) 的形状，这有利于表面重建的平滑和法向量的准确。
        """
        # 获取当前所有高斯球的尺度 (scaling)。注意：通常 scaling 存储的是指数对数或其他激活前的形式，
        # 所以我们需要用对应的激活函数获取实际的缩放系数。
        scales = self.gaussians.get_scaling
        
        # 找到每个高斯球三个轴的缩放比例中最小的那一个（即厚度方向）
        min_scales, _ = torch.min(scales, dim=-1)
        
        # 使其最小尺度趋于 0，通常用 L1 loss
        flat_loss = torch.mean(min_scales)
        
        return flat_loss

    def depth_to_normal(self,view, depth, world_frame=False):
        """
            view: view camera
            depth: depthmap
        """

        points = self.depths_to_points(view, depth, world_frame).reshape(*depth.shape[1:], 3)
        normal_map = torch.zeros_like(points)
        dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
        dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
        normal_map[1:-1, 1:-1, :] = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)


        return normal_map, points

    def depths_to_points(self,view, depthmap, world_frame):
        import math
        W, H = view.image_width, view.image_height
        fx = W / (2 * math.tan(view.FoVx / 2.))
        fy = H / (2 * math.tan(view.FoVy / 2.))
        intrins = torch.tensor([[fx, 0., W / 2.], [0., fy, H / 2.], [0., 0., 1.0]]).float().cuda()
        grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float() + 0.5,
                                        torch.arange(H, device='cuda').float() + 0.5, indexing='xy')
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
        if world_frame:
            c2w = (view.world_view_transform.T).inverse()
            rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
            rays_o = c2w[:3, 3]
            points = depthmap.reshape(-1, 1) * rays_d + rays_o
        else:
            rays_d = points @ intrins.inverse().T
            points = depthmap.reshape(-1, 1) * rays_d
        return points

    def get_w2c_and_depth(self, video_idx, idx, mono_depth,motion_mask, depth_gt,normal,mono, static_mask,print_info=False, init=False):


        est_droid_depth, valid_depth_mask, c2w = self.video.get_depth_and_pose(video_idx, self.device)

        c2w = c2w.to(self.device)
        w2c = torch.linalg.inv(c2w)

        if print_info:
            print(f"valid depth number: {valid_depth_mask.sum().item()}, "
                f"valid depth ratio: {(valid_depth_mask.sum() / (valid_depth_mask.shape[0] * valid_depth_mask.shape[1])).item()}")

        mono_depth_wq = mono_depth

        if valid_depth_mask.sum() < 100:
            invalid = True
            print(
                f"Skip mapping frame {idx} at video idx {video_idx} because of not enough valid depth ({valid_depth_mask.sum()}).")
        else:

            invalid = False
            est_droid_depth[~valid_depth_mask] = 0

            mono_valid_mask = mono_depth < (mono_depth.mean() * 3)
            mono_depth[mono_depth > 3 * mono_depth.mean()] = 0


            from scipy.ndimage import binary_erosion
            mono_depth = mono_depth.cpu().numpy()
            binary_image = (mono_depth > 0).astype(int)
            iterations=5
            padded_binary_image = np.pad(binary_image, pad_width=iterations, mode='constant', constant_values=1)
            structure = np.ones((3, 3), dtype=int) 

    
            eroded_padded_image = binary_erosion(padded_binary_image,
                                                structure=structure,
                                                iterations=iterations)
            eroded_image = eroded_padded_image[iterations:-iterations, iterations:-iterations]


            mono_depth[eroded_image == 0] = 0
    
            if (mono_depth == 0).sum() > 0:
    
                mono_depth = torch.from_numpy(
                    cv2.inpaint(mono_depth,
                                (mono_depth == 0).astype(np.uint8),
                                inpaintRadius=3,  
                                flags=cv2.INPAINT_NS) 
                ).to(self.device)
            else:
                mono_depth = torch.from_numpy(mono_depth).to(self.device)
            normal = normal.cpu().numpy()
            depth_gt = torch.from_numpy(depth_gt).to(self.device)


            if video_idx == self.dystart and len(self.first_d) == 0:
                self.first_d.append(motion_mask)
            if len(self.first_d) == 0:
                print("[Warning] self.first_d is empty, fallback to motion_mask. This may indicate a logic bug.")
                refer_mask = motion_mask
            else:
                refer_mask = self.first_d[0]

            intrinsics = self.frame_reader.get_intrinsic()
            if torch.is_tensor(intrinsics):
                intrinsics = intrinsics.detach().cpu().tolist()
            fx, fy = float(intrinsics[0]), float(intrinsics[1])

            self.obtain_allimgs_st(mono_depth, normal, self.st_predicted, video_idx, fx, fy)
            invalid = 0
            # 与 before_modify 一致：覆盖 running mean，供 align_all_frames 在 s/t 异常时回退
            self.st_predicted["mean_s"] = 0.8
            self.st_predicted["mean_t"] = 0.15

            self.static_msk = motion_mask*refer_mask
            self.align_all_frames(mono_depth, video_idx, self.st_predicted, self.new_scale_alignFrame0, self.static_msk)
            mono_depth = self.export_scaled_pcd(self.st_predicted, self.new_scale_alignFrame0, mono_depth, video_idx,
                                                mean_st=False )
            mono_depth_wq = mono_depth
            torch.cuda.empty_cache()
        return mono_depth_wq, w2c, invalid

    # ---------------------------------------------------------------------
    # Section D: Mapper Render/Deformation and Loss Helper Functions
    # ---------------------------------------------------------------------

    def _unpack_render_pkg(self, render_pkg):
        """Return common render outputs in a fixed order to reduce repeated boilerplate."""
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

    def _mapper_bar_desc(self, label):
        """Match tqdm label style with Printer's [MAPPER] colored prefix."""
        return f"{FontColor.MAPPER}[MAPPER] {Style.RESET_ALL}{label}"

    def _is_reference_viewpoint(self, viewpoint):
        """Use first mapped frame as optimization anchor."""
        if viewpoint is None:
            return False
        if len(self.video_idxs) > 0:
            return int(viewpoint.uid) == int(self.video_idxs[0])
        return int(viewpoint.uid) == 0

    def _get_deform_render_inputs(self, viewpoint, dynamic_network):
        if dynamic_network and self.gaussians.deform_init:
            time_input = self.gaussians.deform.deform.expand_time(viewpoint.fid)
            d_values = self.gaussians.deform.step(
                self.gaussians.get_dygs_xyz.detach(),
                time_input,
                iteration=0,
                feature=None,
                motion_mask=self.gaussians.motion_mask,
                camera_center=viewpoint.camera_center,
                time_interval=self.gaussians.time_interval,
            )
            return (
                d_values["d_xyz"],
                d_values["d_rotation"],
                d_values["d_scaling"],
                d_values["d_opacity"],
                d_values["d_color"],
            )

        return 0, None, 0, None, None

    def _render_with_deform(self, viewpoint, dxyz=0, d_rot=None, d_scale=0, d_opac=None, d_color=None, return_normal=False):
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

    def _get_nearest_view_deform(self, viewpoint):
        closest_keyframe = self.find_closest_keyframe(viewpoint.uid)
        if closest_keyframe is None:
            return None, None

        time_input = self.gaussians.deform.deform.expand_time(
            self.viewpoints[closest_keyframe].fid
        )
        d_value2 = self.gaussians.deform.step(
            self.gaussians.get_dygs_xyz.detach(),
            time_input,
            iteration=0,
            feature=None,
            motion_mask=self.gaussians.motion_mask,
            camera_center=self.viewpoints[closest_keyframe].camera_center,
            time_interval=self.gaussians.time_interval,
        )
        return closest_keyframe, d_value2

    def _sync_refine_keyframes(self):
        for keyframe_idx, frame_idx in zip(self.video_idxs, self.keyframe_idxs):
            _, _, depth_gtd, _, motion_mask, normal, mono, static_msk = self.frame_reader[frame_idx]
            depth_gt_numpy = depth_gtd.cpu().numpy()
            intrinsics = as_intrinsics_matrix(self.frame_reader.get_intrinsic()).to(self.device)
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

            w2c_old = torch.cat((self.cameras[keyframe_idx].R, self.cameras[keyframe_idx].T.unsqueeze(-1)), dim=1)
            w2c_old = torch.cat((w2c_old, torch.tensor([[0, 0, 0, 1]], device="cuda")), dim=0)

            self.cameras[keyframe_idx].update_RT(w2c_temp[:3, :3], w2c_temp[:3, 3])
            self.cameras[keyframe_idx].depth = depth_temp.cpu().numpy()

            if keyframe_idx in self.viewpoints:
                self.viewpoints[keyframe_idx].update_RT(w2c_temp[:3, :3], w2c_temp[:3, 3])
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

    def _compute_flow_pair_loss(self, viewpoint, idx1, dxyz, d_rot, d_scale, flow_weights):
        flow_loss = torch.tensor(0.0, device=self.background.device)
        closest_keyframe = self.find_closest_keyframe(viewpoint.uid)
        if closest_keyframe is None:
            return flow_loss, None, None

        flow, flow_back, _, _, _ = viewpoint.generate_flow(
            viewpoint.original_image.cuda(),
            idx1,
            self.viewpoints[closest_keyframe].original_image.cuda(),
        )

        time_input = self.gaussians.deform.deform.expand_time(
            self.viewpoints[closest_keyframe].fid
        )
        d_value2 = self.gaussians.deform.step(
            self.gaussians.get_dygs_xyz.detach(),
            time_input,
            iteration=0,
            feature=None,
            motion_mask=self.gaussians.motion_mask,
            camera_center=self.viewpoints[closest_keyframe].camera_center,
            time_interval=self.gaussians.time_interval,
        )
        d_xyz2 = d_value2["d_xyz"]

        render_pkg2 = render_flow(
            pc=self.gaussians,
            viewpoint_camera1=viewpoint,
            viewpoint_camera2=self.viewpoints[closest_keyframe],
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
        flow_loss += flow_weights * l1_loss(
            flow_back * dynamic_mask,
            coor1to2_motion * dynamic_mask,
        )

        render_pkg_back = render_flow(
            pc=self.gaussians,
            viewpoint_camera1=self.viewpoints[closest_keyframe],
            viewpoint_camera2=viewpoint,
            d_xyz1=d_xyz2,
            d_xyz2=dxyz,
            d_rotation1=d_value2["d_rotation"],
            d_scaling1=d_value2["d_scaling"],
        )
        coor2to1_motion = render_pkg_back["render"][:2].permute(1, 2, 0)
        dynamic_mask = (
            (~self.viewpoints[closest_keyframe].motion_mask)
            .unsqueeze(0)
            .permute(1, 2, 0)
            .repeat(1, 1, 2)
            .detach()
        )
        flow_loss += flow_weights * l1_loss(
            flow * dynamic_mask,
            coor2to1_motion * dynamic_mask,
        )

        return flow_loss, closest_keyframe, d_value2

    def _compute_dynamic_mapping_loss(
        self,
        viewpoint,
        image,
        depth,
        opacity,
        render_pkg,
        dynamic,
        dynamic_network,
        closest_keyframe,
        d_value2,
    ):
        order_mask = viewpoint.depth > 0
        loss_order_depth = self.get_depth_order_loss(depth, viewpoint.depth, order_mask)
        loss_mapping = get_loss_mapping(
            self.config["mapping"],
            image,
            depth,
            viewpoint,
            opacity,
            rm_dynamic=not dynamic_network,
            dynamic=dynamic,
        ) + self.depth_order_loss_weight * loss_order_depth
        loss_mapping += self.normal_loss_weight * self.get_loss_normal(depth, viewpoint)

        enable_multi_view = not dynamic
        
        if self.use_normal:
            render_normal = render_pkg.get("normal", None)
            if render_normal is not None:
                loss_mapping += self.config["mapping"]["Training"]["render_normal_loss_weight"] * self.get_render_normal_loss(viewpoint.normal, render_normal)
                loss_mapping += self.config["mapping"]["Training"]["flat_loss_weight"] * self.get_flat_loss()

        if (
            # False and
            self.config["mapping"]["Training"]["multi_loss_weight"] and
            enable_multi_view
            and closest_keyframe is not None
            and closest_keyframe in self.viewpoints
            and d_value2 is not None
        ):
            loss_mapping += self.config["mapping"]["Training"]["multi_loss_weight"] * self.compute_multi_view_loss(
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

        return loss_mapping

    def _compute_deform_reg_loss(self, viewpoint, delta, strong_reg):
        reg_loss = torch.tensor(0.0, device=self.background.device)
        if strong_reg:
            reg_loss += 1e-3 * self.gaussians.deform.deform.arap_loss(
                t=viewpoint.fid,
                delta_t=delta * self.gaussians.time_interval,
                t_samp_num=4,
            )
            reg_loss += 1e-5 * self.gaussians.deform.deform.acc_loss(
                t=viewpoint.fid,
                delta_t=5 * self.gaussians.time_interval,
            )
            reg_loss += 1e-3 * self.gaussians.deform.deform.elastic_loss(
                t=viewpoint.fid,
                delta_t=5 * self.gaussians.time_interval,
            )
            return reg_loss

        reg_loss += 1e-4 * self.gaussians.deform.deform.elastic_loss(
            t=viewpoint.fid,
            delta_t=5 * self.gaussians.time_interval,
        )
        reg_loss += 1e-6 * self.gaussians.deform.deform.acc_loss(
            t=viewpoint.fid,
            delta_t=5 * self.gaussians.time_interval,
        )
        reg_loss += 1e-4 * self.gaussians.deform.deform.arap_loss(
            t=viewpoint.fid,
            delta_t=5 * self.gaussians.time_interval,
        )
        return reg_loss

    def _collect_window_psnr(self, window_indices, dynamic_network):
        from thirdparty.gaussian_splatting.utils.image_utils import psnr

        valid_indices = [int(kf_idx) for kf_idx in window_indices if int(kf_idx) in self.viewpoints]
        valid_indices = sorted(valid_indices)
        if len(valid_indices) == 0:
            return float("nan"), []

        psnr_items = []
        with torch.no_grad():
            for kf_idx in valid_indices:
                viewpoint = self.viewpoints[kf_idx]
                dxyz, d_rot, d_scale, d_opac, d_color = self._get_deform_render_inputs(
                    viewpoint,
                    dynamic_network,
                )
                render_pkg = render(
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
                )
                image = torch.clamp(render_pkg["render"].detach(), 0.0, 1.0)
                gt_image = viewpoint.original_image

                if hasattr(self, "video_idxs") and len(self.video_idxs) > 0 and viewpoint.uid != self.video_idxs[0]:
                    image = (torch.exp(viewpoint.exposure_a.detach())) * image + viewpoint.exposure_b.detach()
                    image = torch.clamp(image, 0.0, 1.0)

                mask = gt_image > 0
                if mask.any():
                    psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0)).item()
                    psnr_items.append((int(viewpoint.uid), float(psnr_score)))

        if len(psnr_items) == 0:
            return float("nan"), []

        avg_psnr = float(np.mean([x[1] for x in psnr_items]))
        return avg_psnr, psnr_items

    def _log_window_optimization_stats(self, stage_tag, window_indices, dynamic_network):
        avg_psnr, psnr_items = self._collect_window_psnr(window_indices, dynamic_network)

        total_gaussians = int(self.gaussians.get_xyz.shape[0])
        dynamic_gaussians = 0
        if hasattr(self.gaussians, "dygs") and torch.is_tensor(self.gaussians.dygs) and self.gaussians.dygs.numel() > 0:
            dynamic_gaussians = int(self.gaussians.dygs.long().sum().item())

        valid_indices = [int(kf_idx) for kf_idx in window_indices if int(kf_idx) in self.viewpoints]
        valid_indices = sorted(valid_indices)
        if len(valid_indices) > 0:
            window_desc = f"len={len(valid_indices)}, range=[{valid_indices[0]}->{valid_indices[-1]}]"
        else:
            window_desc = "len=0"

        if len(psnr_items) > 0:
            psnr_detail = ", ".join([f"{uid}:{score:.2f}" for uid, score in psnr_items])
            msg = (
                f"[{stage_tag}] window({window_desc}) | avg_psnr={avg_psnr:.2f} | "
                f"psnr[{psnr_detail}] | gaussians={total_gaussians} | dynamic_gaussians={dynamic_gaussians}"
            )
        else:
            msg = (
                f"[{stage_tag}] window({window_desc}) | avg_psnr=N/A | "
                f"gaussians={total_gaussians} | dynamic_gaussians={dynamic_gaussians}"
            )

        self.printer.print(msg, FontColor.MAPPER)

        if self.keep_dynamic_gaussians:
            if self._last_dynamic_gaussians is not None and dynamic_gaussians < self._last_dynamic_gaussians:
                warn_msg = (
                    f"[{stage_tag}] dynamic_gaussians decreased: "
                    f"{self._last_dynamic_gaussians} -> {dynamic_gaussians}. "
                    "Check dygs flags or additional prune paths."
                )
                self.printer.print(warn_msg, FontColor.MAPPER)
            self._last_dynamic_gaussians = dynamic_gaussians

    # ---------------------------------------------------------------------
    # Section E: Online Visualization Utilities
    # ---------------------------------------------------------------------

    @staticmethod
    def _chw01_to_rgb_uint8_hwc(img_t):
        """CHW float [0,1] tensor -> HWC uint8 RGB (dataset/render are RGB, not OpenCV BGR)."""
        arr = img_t.detach().cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
        return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)

    def _depth_l1_diff_for_online_plot(self, pred_depth, gt_depth):
        """Aligned abs depth diff (masked) and mean L1 on valid gt depth; avoids numpy/torch mix & div-by-zero."""
        d = pred_depth.detach().cpu().float()
        if d.dim() == 2:
            d = d.unsqueeze(0)
        gt = torch.as_tensor(np.asarray(gt_depth), dtype=torch.float32)
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)
        diff = torch.abs(d - gt)
        valid = gt > 0
        denom = valid.float().sum().clamp(min=1.0)
        depth_l1 = (diff * valid.float()).sum() / denom
        return diff * valid.float(), depth_l1

    def _plot_window_psnr_grid(self, window_indices, dynamic_network, plot_dir, idx_tag):
        from thirdparty.gaussian_splatting.utils.image_utils import psnr

        valid_indices = [int(kf_idx) for kf_idx in window_indices if int(kf_idx) in self.viewpoints]
        valid_indices = sorted(valid_indices)
        if len(valid_indices) == 0:
            return

        rows, cols = 2, 4
        max_slots = rows * cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        axes = np.array(axes, ndmin=2).reshape(rows, cols)

        for axis in axes.flat:
            axis.axis("off")

        with torch.no_grad():
            for plot_idx, kf_idx in enumerate(valid_indices[:max_slots]):
                row = plot_idx // cols
                col = plot_idx % cols
                axis = axes[row, col]
                viewpoint = self.viewpoints[kf_idx]

                dxyz, d_rot, d_scale, d_opac, d_color = self._get_deform_render_inputs(
                    viewpoint,
                    dynamic_network,
                )
                render_pkg = render(
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
                )
                image = torch.clamp(render_pkg["render"].detach(), 0.0, 1.0)
                gt_image = viewpoint.original_image

                if hasattr(self, "video_idxs") and len(self.video_idxs) > 0 and viewpoint.uid != self.video_idxs[0]:
                    image = (torch.exp(viewpoint.exposure_a.detach())) * image + viewpoint.exposure_b.detach()
                    image = torch.clamp(image, 0.0, 1.0)

                mask = gt_image > 0
                if mask.any():
                    psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0)).item()
                    title = f"uid {viewpoint.uid} | PSNR {psnr_score:.2f}"
                else:
                    title = f"uid {viewpoint.uid} | PSNR N/A"

                axis.imshow(image.cpu().permute(1, 2, 0))
                axis.set_title(title)

        window_plot_dir = os.path.join(plot_dir, "window_psnr")
        os.makedirs(window_plot_dir, exist_ok=True)
        save_path = os.path.join(window_plot_dir, f"{idx_tag}.png")
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    def _normal_to_rgb(self, normal_map):
        """Convert normal map to RGB visualization in [0, 1]."""
        if normal_map is None:
            return None

        nm = normal_map.detach()
        if nm.dim() == 4:
            nm = nm.squeeze(0)

        if nm.dim() == 3 and nm.shape[0] == 3:
            nm = nm.permute(1, 2, 0)
        elif nm.dim() == 3 and nm.shape[-1] == 3:
            pass
        else:
            return None

        nm = torch.nn.functional.normalize(nm, dim=-1)
        rgb = (nm + 1.0) * 0.5
        return torch.clamp(rgb, 0.0, 1.0)

    def _plot_window_normal_grid(self, window_indices, dynamic_network, plot_dir, idx_tag):
        valid_indices = [int(kf_idx) for kf_idx in window_indices if int(kf_idx) in self.viewpoints]
        valid_indices = sorted(valid_indices)
        if len(valid_indices) == 0:
            return

        cols = min(4, len(valid_indices))
        rows = (len(valid_indices) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        axes = np.array(axes, ndmin=2).reshape(rows, cols)

        for axis in axes.flat:
            axis.axis("off")

        with torch.no_grad():
            for plot_idx, kf_idx in enumerate(valid_indices):
                row = plot_idx // cols
                col = plot_idx % cols
                axis = axes[row, col]
                viewpoint = self.viewpoints[kf_idx]

                dxyz, d_rot, d_scale, d_opac, d_color = self._get_deform_render_inputs(
                    viewpoint,
                    dynamic_network,
                )
                render_pkg = self._render_with_deform(
                    viewpoint,
                    dxyz=dxyz,
                    d_rot=d_rot,
                    d_scale=d_scale,
                    d_opac=d_opac,
                    d_color=d_color,
                    return_normal=self.use_normal,
                )
                render_normal = render_pkg.get("normal", None)
                render_rgb = self._normal_to_rgb(render_normal)
                if render_rgb is None:
                    axis.set_title(f"uid {viewpoint.uid} | normal N/A")
                    continue

                title = f"uid {viewpoint.uid}"
                gt_normal = getattr(viewpoint, "normal", None)
                gt_rgb = self._normal_to_rgb(gt_normal)
                if gt_rgb is not None and gt_rgb.shape == render_rgb.shape:
                    gt_nm = torch.nn.functional.normalize(gt_rgb * 2.0 - 1.0, dim=-1)
                    pred_nm = torch.nn.functional.normalize(render_rgb * 2.0 - 1.0, dim=-1)
                    cos_sim = (gt_nm * pred_nm).sum(dim=-1).clamp(-1.0, 1.0)
                    valid_mask = gt_nm.norm(dim=-1) > 0.1
                    if valid_mask.any():
                        ang = torch.rad2deg(torch.acos(cos_sim[valid_mask])).mean().item()
                        title = f"uid {viewpoint.uid} | Nerr {ang:.2f}deg"

                axis.imshow(render_rgb.cpu().numpy())
                axis.set_title(title)

        window_plot_dir = os.path.join(plot_dir, "window_normal")
        os.makedirs(window_plot_dir, exist_ok=True)
        save_path = os.path.join(window_plot_dir, f"{idx_tag}.png")
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    def _overlay_dynamic_mask(self, axis, viewpoint, height, width, alpha=0.28):
        motion_mask = getattr(viewpoint, "motion_mask", None)
        if motion_mask is None:
            return

        static_mask = motion_mask.detach() if torch.is_tensor(motion_mask) else torch.from_numpy(np.asarray(motion_mask))
        if static_mask.dim() == 3:
            static_mask = static_mask.squeeze(0)
        static_mask = static_mask.bool()

        if static_mask.shape[0] != height or static_mask.shape[1] != width:
            static_mask = F.interpolate(
                static_mask.float().unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode="nearest",
            ).squeeze(0).squeeze(0).bool()

        dynamic_mask = ~static_mask
        if not dynamic_mask.any():
            return

        dynamic_np = dynamic_mask.detach().cpu().numpy()
        overlay = np.zeros((height, width, 4), dtype=np.float32)
        # Red transparent overlay for dynamic region.
        overlay[..., 0] = 1.0
        overlay[..., 1] = 0.15
        overlay[..., 2] = 0.12
        overlay[..., 3] = dynamic_np.astype(np.float32) * float(alpha)
        axis.imshow(overlay)

    def _get_dynamic_gaussian_trajectory(self, viewpoint, t_samp_num):
        if not self.dynamic_model or not self.gaussians.deform_init:
            return None

        deform_model = getattr(self.gaussians, "deform", None)
        if deform_model is None or not hasattr(deform_model, "step"):
            return None

        base_xyz_all = self.gaussians.get_dygs_xyz.detach()
        if base_xyz_all.numel() == 0:
            return None

        n_dy = base_xyz_all.shape[0]
        if self.node_traj_max_nodes > 0 and n_dy > self.node_traj_max_nodes:
            if self.node_traj_subset_seed >= 0:
                g = torch.Generator(device=base_xyz_all.device)
                g.manual_seed(int(self.node_traj_subset_seed))
                perm = torch.randperm(n_dy, generator=g, device=base_xyz_all.device)[
                    : self.node_traj_max_nodes
                ]
                sample_idx = perm.sort().values
            else:
                sample_idx = torch.linspace(
                    0,
                    n_dy - 1,
                    steps=self.node_traj_max_nodes,
                    device=base_xyz_all.device,
                ).long()
            base_xyz = base_xyz_all[sample_idx]
        else:
            base_xyz = base_xyz_all

        t_samp_num = max(2, int(t_samp_num))
        ts = torch.linspace(0.0, 1.0, t_samp_num, device=base_xyz.device, dtype=base_xyz.dtype)
        sampled_motion_mask = torch.ones_like(base_xyz[:, :1])

        traj_list = []
        for t in ts:
            time_input = self.gaussians.deform.deform.expand_time(t)
            d_values = self.gaussians.deform.step(
                base_xyz,
                time_input,
                iteration=0,
                feature=None,
                motion_mask=sampled_motion_mask,
                camera_center=viewpoint.camera_center,
                time_interval=self.gaussians.time_interval,
            )
            d_xyz = d_values.get("d_xyz", None)
            if d_xyz is None:
                cur_xyz = base_xyz
            else:
                cur_xyz = base_xyz + d_xyz
            traj_list.append(cur_xyz.unsqueeze(1))

        return torch.cat(traj_list, dim=1)

    def _compute_dynamic_tracks_for_views(self, view_seq):
        if len(view_seq) < 2:
            return None

        base_xyz_all = self.gaussians.get_dygs_xyz.detach()
        if base_xyz_all.numel() == 0:
            return None

        n_dy = base_xyz_all.shape[0]
        if self.node_traj_max_nodes > 0 and n_dy > self.node_traj_max_nodes:
            if self.node_traj_subset_seed >= 0:
                g = torch.Generator(device=base_xyz_all.device)
                g.manual_seed(int(self.node_traj_subset_seed))
                perm = torch.randperm(n_dy, generator=g, device=base_xyz_all.device)[
                    : self.node_traj_max_nodes
                ]
                sample_idx = perm.sort().values
            else:
                sample_idx = torch.linspace(
                    0,
                    n_dy - 1,
                    steps=self.node_traj_max_nodes,
                    device=base_xyz_all.device,
                ).long()
            base_xyz = base_xyz_all[sample_idx]
        else:
            base_xyz = base_xyz_all

        if base_xyz.shape[0] == 0:
            return None

        traj_world = []
        uv_seq = []
        valid_seq = []
        frame_seq = []
        dynamic_mask_seq = []
        view_ids = []

        motion_mask_sample = torch.ones_like(base_xyz[:, :1])
        for view in view_seq:
            time_input = self.gaussians.deform.deform.expand_time(view.fid)
            d_values = self.gaussians.deform.step(
                base_xyz,
                time_input,
                iteration=0,
                feature=None,
                motion_mask=motion_mask_sample,
                camera_center=view.camera_center,
                time_interval=self.gaussians.time_interval,
            )
            d_xyz = d_values.get("d_xyz", None)
            cur_xyz = base_xyz if d_xyz is None else (base_xyz + d_xyz)
            traj_world.append(cur_xyz)

            pts4 = torch.cat(
                [
                    cur_xyz,
                    torch.ones(
                        cur_xyz.shape[0],
                        1,
                        device=cur_xyz.device,
                        dtype=cur_xyz.dtype,
                    ),
                ],
                dim=1,
            )
            pts_cam_h = pts4 @ view.world_view_transform
            pts_cam = pts_cam_h[:, :3]
            z = pts_cam[:, 2]
            u = pts_cam[:, 0] * view.fx / z.clamp_min(1e-6) + view.cx
            v = pts_cam[:, 1] * view.fy / z.clamp_min(1e-6) + view.cy
            uv = torch.stack([u, v], dim=-1)

            img = torch.clamp(view.original_image.detach(), 0.0, 1.0)
            img_np = img.permute(1, 2, 0).cpu().numpy()
            h, w = img_np.shape[:2]
            in_img = (
                (uv[:, 0] >= 0)
                & (uv[:, 0] < w)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < h)
            )
            valid = (z > 1e-6) & in_img

            dyn_mask = None
            if hasattr(view, "motion_mask") and view.motion_mask is not None:
                static_mask = view.motion_mask.detach() if torch.is_tensor(view.motion_mask) else torch.from_numpy(np.asarray(view.motion_mask))
                if static_mask.dim() == 3:
                    static_mask = static_mask.squeeze(0)
                static_mask = static_mask.bool()
                if static_mask.shape[0] != h or static_mask.shape[1] != w:
                    static_mask = F.interpolate(
                        static_mask.float().unsqueeze(0).unsqueeze(0),
                        size=(h, w),
                        mode="nearest",
                    ).squeeze(0).squeeze(0).bool()
                dyn_mask = ~static_mask
                uv_round = uv.round().long()
                uv_round[:, 0] = uv_round[:, 0].clamp(0, w - 1)
                uv_round[:, 1] = uv_round[:, 1].clamp(0, h - 1)
                valid = valid & dyn_mask[uv_round[:, 1], uv_round[:, 0]]

            uv_seq.append(uv.detach().cpu().numpy())
            valid_seq.append(valid.detach().cpu().numpy())
            frame_seq.append(img_np)
            dynamic_mask_seq.append(None if dyn_mask is None else dyn_mask.detach().cpu().numpy())
            view_ids.append(int(view.uid))

        if len(frame_seq) < 2:
            return None

        return {
            "traj_world": torch.stack(traj_world, dim=1).detach().cpu().numpy(),  # [N, T, 3]
            "uv": np.stack(uv_seq, axis=0),  # [T, N, 2]
            "valid": np.stack(valid_seq, axis=0),  # [T, N]
            "frames": frame_seq,
            "dynamic_masks": dynamic_mask_seq,
            "view_ids": np.asarray(view_ids, dtype=np.int32),
            "num_points": int(base_xyz.shape[0]),
        }

    def _plot_window_dynamic_trajectory_2d_grid(
        self, window_indices, dynamic_network, plot_dir, idx_tag, allow_without_online_vis=False
    ):
        if not (dynamic_network and self.gaussians.deform_init):
            return
        if not allow_without_online_vis and not self.node_traj_vis:
            return

        valid_indices = [int(kf_idx) for kf_idx in window_indices if int(kf_idx) in self.viewpoints]
        valid_indices = sorted(valid_indices)
        if len(valid_indices) < 2:
            return

        view_seq = [self.viewpoints[k] for k in valid_indices]
        payload = self._compute_dynamic_tracks_for_views(view_seq)
        if payload is None:
            return

        uv_arr = payload["uv"]
        valid_arr = payload["valid"]
        frame_seq = payload["frames"]
        dynamic_mask_seq = payload["dynamic_masks"]
        view_ids = payload["view_ids"]
        num_points = payload["num_points"]

        rows, cols = 2, 4
        max_slots = rows * cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        axes = np.array(axes, ndmin=2).reshape(rows, cols)
        for axis in axes.flat:
            axis.axis("off")

        colors = plt.cm.turbo(np.linspace(0.0, 1.0, num_points))
        for plot_idx in range(min(len(frame_seq), max_slots)):
            row = plot_idx // cols
            col = plot_idx % cols
            axis = axes[row, col]

            frame = frame_seq[plot_idx]
            axis.imshow(frame)
            h, w = frame.shape[:2]

            dyn_mask = dynamic_mask_seq[plot_idx]
            if dyn_mask is not None:
                overlay = np.zeros((h, w, 4), dtype=np.float32)
                overlay[..., 0] = 1.0
                overlay[..., 1] = 0.15
                overlay[..., 2] = 0.12
                overlay[..., 3] = dyn_mask.astype(np.float32) * 0.24
                axis.imshow(overlay)

            for p_idx in range(num_points):
                pts = []
                for t_idx in range(plot_idx + 1):
                    if valid_arr[t_idx, p_idx]:
                        u, v = uv_arr[t_idx, p_idx]
                        pts.append((float(u), float(v)))

                if len(pts) >= 2:
                    c = np.asarray(pts, dtype=np.float32)
                    axis.plot(c[:, 0], c[:, 1], color=colors[p_idx], linewidth=0.9, alpha=0.75)
                if len(pts) >= 1:
                    axis.scatter(pts[-1][0], pts[-1][1], color=colors[p_idx], s=7, alpha=0.9)

            axis.set_title(f"uid {view_ids[plot_idx]} | dyn tracks")
            axis.set_xlim([0, w])
            axis.set_ylim([h, 0])

        node_plot_dir = os.path.join(plot_dir, "node_trajectory")
        os.makedirs(node_plot_dir, exist_ok=True)
        save_path = os.path.join(node_plot_dir, f"{idx_tag}_2d_grid.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def _plot_window_dynamic_trajectory_3d(
        self, window_indices, dynamic_network, plot_dir, idx_tag, allow_without_online_vis=False
    ):
        if not (dynamic_network and self.gaussians.deform_init):
            return
        if not allow_without_online_vis and not self.node_traj_vis:
            return

        valid_indices = [int(kf_idx) for kf_idx in window_indices if int(kf_idx) in self.viewpoints]
        valid_indices = sorted(valid_indices)
        if len(valid_indices) < 2:
            return

        view_seq = [self.viewpoints[k] for k in valid_indices]
        payload = self._compute_dynamic_tracks_for_views(view_seq)
        if payload is None:
            return

        traj_world = payload["traj_world"]
        view_ids = payload["view_ids"]
        num_points = payload["num_points"]

        fig = plt.figure(figsize=(10, 8))
        ax3d = fig.add_subplot(1, 1, 1, projection="3d")
        colors = plt.cm.turbo(np.linspace(0.0, 1.0, num_points))

        for p_idx in range(num_points):
            xyz = traj_world[p_idx]
            ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=colors[p_idx], linewidth=0.9, alpha=0.8)
            ax3d.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], color=colors[p_idx], s=6, alpha=0.9)

        if allow_without_online_vis:
            all_xyz = traj_world.reshape(-1, 3)
            cmin = all_xyz.min(axis=0)
            cmax = all_xyz.max(axis=0)
            center = 0.5 * (cmin + cmax)
            span = float(np.max(cmax - cmin)) + 1e-6
            half = 0.5 * span * (1.0 + self.node_traj_3d_margin)
            ax3d.set_xlim(center[0] - half, center[0] + half)
            ax3d.set_ylim(center[1] - half, center[1] + half)
            ax3d.set_zlim(center[2] - half, center[2] + half)
            try:
                ax3d.set_box_aspect((1, 1, 1))
            except Exception:
                pass
            ax3d.view_init(elev=self.node_traj_3d_elev, azim=self.node_traj_3d_azim_start)
            try:
                ax3d.dist = 10.5
            except Exception:
                pass

        ax3d.set_title(
            f"Dynamic Gaussian Center 3D Motion | window [{view_ids[0]}->{view_ids[-1]}] | points {num_points}"
        )
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")

        node_plot_dir = os.path.join(plot_dir, "node_trajectory")
        os.makedirs(node_plot_dir, exist_ok=True)
        save_path = os.path.join(node_plot_dir, f"{idx_tag}_3d.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _matplotlib_fig_to_rgb_uint8(self, fig, dpi=110):
        import io

        import imageio.v2 as imageio

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        arr = imageio.imread(buf)
        plt.close(fig)
        return arr

    def _save_node_traj_gif_2d_grid(self, payload, node_plot_dir, idx_tag, fps):
        """按关键帧时间展开 2x4 网格：第 s 帧仅显示前 s+1 个视角上的累积轨迹，并写成 GIF。"""
        import imageio.v2 as imageio

        uv_arr = payload["uv"]
        valid_arr = payload["valid"]
        frame_seq = payload["frames"]
        dynamic_mask_seq = payload["dynamic_masks"]
        view_ids = payload["view_ids"]
        num_points = payload["num_points"]
        T = len(frame_seq)
        if T < 2:
            return

        step_indices = list(range(T))
        max_f = self.node_traj_gif_max_frames
        if max_f > 0 and len(step_indices) > max_f:
            step_indices = np.linspace(0, T - 1, num=max_f, dtype=int).tolist()

        rows, cols = 2, 4
        max_slots = rows * cols
        colors = plt.cm.turbo(np.linspace(0.0, 1.0, num_points))
        frames_rgb = []

        for s in step_indices:
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
            axes = np.array(axes, ndmin=2).reshape(rows, cols)
            for axis in axes.flat:
                axis.axis("off")

            for plot_idx in range(min(len(frame_seq), max_slots)):
                row = plot_idx // cols
                col = plot_idx % cols
                axis = axes[row, col]
                if plot_idx > s:
                    axis.set_facecolor("#2a2a2a")
                    axis.text(
                        0.5,
                        0.5,
                        "…",
                        transform=axis.transAxes,
                        ha="center",
                        va="center",
                        color="w",
                        fontsize=16,
                    )
                    continue

                frame = frame_seq[plot_idx]
                axis.imshow(frame)
                h, w = frame.shape[:2]

                dyn_mask = dynamic_mask_seq[plot_idx]
                if dyn_mask is not None:
                    overlay = np.zeros((h, w, 4), dtype=np.float32)
                    overlay[..., 0] = 1.0
                    overlay[..., 1] = 0.15
                    overlay[..., 2] = 0.12
                    overlay[..., 3] = dyn_mask.astype(np.float32) * 0.24
                    axis.imshow(overlay)

                for p_idx in range(num_points):
                    pts = []
                    for t_idx in range(plot_idx + 1):
                        if valid_arr[t_idx, p_idx]:
                            u, v = uv_arr[t_idx, p_idx]
                            pts.append((float(u), float(v)))

                    if len(pts) >= 2:
                        c = np.asarray(pts, dtype=np.float32)
                        axis.plot(
                            c[:, 0],
                            c[:, 1],
                            color=colors[p_idx],
                            linewidth=0.9,
                            alpha=0.75,
                        )
                    if len(pts) >= 1:
                        axis.scatter(
                            pts[-1][0],
                            pts[-1][1],
                            color=colors[p_idx],
                            s=7,
                            alpha=0.9,
                        )

                axis.set_title(f"uid {view_ids[plot_idx]} | t≤{s}")
                axis.set_xlim([0, w])
                axis.set_ylim([h, 0])

            fig.suptitle(
                f"Dynamic tracks 2D (subset n={num_points}) | frame {s + 1}/{T}",
                fontsize=11,
            )
            fig.tight_layout()
            frames_rgb.append(self._matplotlib_fig_to_rgb_uint8(fig, dpi=110))

        out_path = os.path.join(node_plot_dir, f"{idx_tag}_2d_grid.gif")
        os.makedirs(node_plot_dir, exist_ok=True)
        imageio.mimsave(out_path, frames_rgb, fps=max(1, int(fps)), loop=0)
        self.printer.print(f"Saved node trajectory 2D GIF: {out_path}", FontColor.MAPPER)

    def _save_node_traj_gif_3d(self, payload, node_plot_dir, idx_tag, fps):
        """世界系下轨迹随时间生长；相机略远、立方体包围盒，方位角缓慢旋转以观察全场景与运动。"""
        import imageio.v2 as imageio

        traj_world = payload["traj_world"]
        view_ids = payload["view_ids"]
        num_points = payload["num_points"]
        T = traj_world.shape[1]
        if T < 2:
            return

        step_indices = list(range(T))
        max_f = self.node_traj_gif_max_frames
        if max_f > 0 and len(step_indices) > max_f:
            step_indices = np.linspace(0, T - 1, num=max_f, dtype=int).tolist()

        all_xyz = traj_world.reshape(-1, 3)
        cmin = all_xyz.min(axis=0)
        cmax = all_xyz.max(axis=0)
        center = 0.5 * (cmin + cmax)
        span = float(np.max(cmax - cmin)) + 1e-6
        half = 0.5 * span * (1.0 + self.node_traj_3d_margin)

        colors = plt.cm.turbo(np.linspace(0.0, 1.0, num_points))
        frames_rgb = []

        for s in step_indices:
            fig = plt.figure(figsize=(11, 9))
            ax3d = fig.add_subplot(1, 1, 1, projection="3d")
            for p_idx in range(num_points):
                seg = traj_world[p_idx, : s + 1, :]
                ax3d.plot(
                    seg[:, 0],
                    seg[:, 1],
                    seg[:, 2],
                    color=colors[p_idx],
                    linewidth=1.05,
                    alpha=0.88,
                )
                ax3d.scatter(
                    seg[-1, 0],
                    seg[-1, 1],
                    seg[-1, 2],
                    color=colors[p_idx],
                    s=10,
                    alpha=0.95,
                )

            ax3d.set_xlim(center[0] - half, center[0] + half)
            ax3d.set_ylim(center[1] - half, center[1] + half)
            ax3d.set_zlim(center[2] - half, center[2] + half)
            try:
                ax3d.set_box_aspect((1, 1, 1))
            except Exception:
                pass
            azim = self.node_traj_3d_azim_start + self.node_traj_3d_azim_spin * (
                float(s) / float(max(T - 1, 1))
            )
            ax3d.view_init(elev=self.node_traj_3d_elev, azim=azim)
            try:
                ax3d.dist = 10.5
            except Exception:
                pass

            ax3d.set_title(
                f"Dynamic Gaussians 3D | [{view_ids[0]}→{view_ids[-1]}] | n={num_points} | step {s + 1}/{T}",
                fontsize=11,
            )
            ax3d.set_xlabel("X")
            ax3d.set_ylabel("Y")
            ax3d.set_zlabel("Z")

            fig.tight_layout()
            frames_rgb.append(self._matplotlib_fig_to_rgb_uint8(fig, dpi=115))

        out_path = os.path.join(node_plot_dir, f"{idx_tag}_3d.gif")
        os.makedirs(node_plot_dir, exist_ok=True)
        imageio.mimsave(out_path, frames_rgb, fps=max(1, int(fps)), loop=0)
        self.printer.print(f"Saved node trajectory 3D GIF: {out_path}", FontColor.MAPPER)

    def _plot_node_trajectory(
        self,
        viewpoint,
        dynamic_network,
        plot_dir,
        idx_tag,
        t_samp_num_override=None,
        force_t_idx=None,
        draw_history=True,
    ):
        del t_samp_num_override, force_t_idx, draw_history
        if not (self.node_traj_vis and dynamic_network and self.gaussians.deform_init):
            return
        if self.node_traj_vis_every > 1 and int(viewpoint.uid) % self.node_traj_vis_every != 0:
            return

        window_indices = self.current_window if len(self.current_window) > 0 else [int(viewpoint.uid)]
        self._plot_window_dynamic_trajectory_2d_grid(window_indices, dynamic_network, plot_dir, idx_tag)
        self._plot_window_dynamic_trajectory_3d(window_indices, dynamic_network, plot_dir, idx_tag)

    def _plot_node_trajectory_full_sequence(self):
        if not self.node_traj_final_vis:
            return
        if not self.dynamic_model or not self.gaussians.deform_init:
            return
        if len(self.viewpoints) < 2:
            return

        view_ids = sorted(self.viewpoints.keys())
        plot_dir = os.path.join(self.save_dir, "online_plots")
        self._plot_window_dynamic_trajectory_2d_grid(
            view_ids, True, plot_dir, "full_sequence", allow_without_online_vis=True
        )
        self._plot_window_dynamic_trajectory_3d(
            view_ids, True, plot_dir, "full_sequence", allow_without_online_vis=True
        )
        if self.node_traj_gif:
            valid_indices = [int(k) for k in view_ids if int(k) in self.viewpoints]
            valid_indices = sorted(valid_indices)
            view_seq = [self.viewpoints[k] for k in valid_indices]
            payload = self._compute_dynamic_tracks_for_views(view_seq)
            if payload is not None:
                node_plot_dir = os.path.join(plot_dir, "node_trajectory")
                self._save_node_traj_gif_2d_grid(
                    payload, node_plot_dir, "full_sequence", self.node_traj_gif_fps
                )
                self._save_node_traj_gif_3d(
                    payload, node_plot_dir, "full_sequence", self.node_traj_gif_fps
                )

    # ---------------------------------------------------------------------
    # Section F: Map Initialization and Mapping Core Loop
    # ---------------------------------------------------------------------


    def initialize_map(self, cur_frame_idx,idx, viewpoint):
        # Mapping stage: render current view, build initialization loss, then optimize gaussians.
        for mapping_iteration in tqdm(
            range(self.init_itr_num),
            desc=self._mapper_bar_desc(f"init map[{idx}]"),
            position=0,
            dynamic_ncols=True,
            leave=True,
        ):
            self.iteration_count += 1
            render_pkg = self._render_with_deform(viewpoint, return_normal=self.use_normal)
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
                normal
            ) = self._unpack_render_pkg(render_pkg)
            loss_init = get_loss_mapping(
                self.config["mapping"], image, depth, viewpoint, opacity, initialization=True,rm_dynamic=not (self.dystart==cur_frame_idx)
            )
            if self.use_normal:
                loss_init += self.config["mapping"]["Training"]["render_normal_loss_weight"] * self.get_render_normal_loss(viewpoint.normal, normal)
                loss_init += self.config["mapping"]["Training"]["flat_loss_weight"] * self.get_flat_loss()
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    init_dynamic_prune_gate = None
                    if (
                        self.preserve_invisible_gaussians
                        and hasattr(self.gaussians, "dygs")
                        and torch.is_tensor(self.gaussians.dygs)
                        and self.gaussians.dygs.numel() == visibility_filter.numel()
                    ):
                        dynamic_mask = self.gaussians.dygs.bool().to(visibility_filter.device)
                        init_dynamic_prune_gate = torch.where(
                            dynamic_mask,
                            visibility_filter,
                            torch.ones_like(visibility_filter, dtype=torch.bool),
                        )
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                        visible_mask=init_dynamic_prune_gate,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                        self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        vis_render_process(self.gaussians, self.pipeline_params, self.background, viewpoint,
                        viewpoint.uid, self.save_dir, out_dir="map", mask=None, dynamic=False)
        self.printer.print("Initialized map", FontColor.MAPPER)
        stats_window = self.current_window if len(self.current_window) > 0 else [int(cur_frame_idx)]
        self._log_window_optimization_stats("init_map", stats_window, False)

        # online plotting
        if self.online_plotting:
            from thirdparty.gaussian_splatting.utils.image_utils import psnr
            from src.utils.eval_utils import plot_rgbd_silhouette
            cur_idx = self.current_window[np.array(self.current_window).argmax()]
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

            if len(self.video_idxs) > 0 and viewpoint.uid != self.video_idxs[0]:
                image = (torch.exp(viewpoint.exposure_a.detach())) * image + viewpoint.exposure_b.detach()
            image = torch.clamp(image, 0.0, 1.0)
            gt = self._chw01_to_rgb_uint8_hwc(gt_image)
            pred = self._chw01_to_rgb_uint8_hwc(image)
            mask = gt_image > 0
            psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
            diff_depth_l1, depth_l1 = self._depth_l1_diff_for_online_plot(depth, gt_depth)

            plot_dir = os.path.join(self.save_dir, "online_plots")
            plot_rgbd_silhouette(gt_image, gt_depth, image, depth, diff_depth_l1,
                                psnr_score.item(), depth_l1.item(), plot_dir=plot_dir, idx=str(cur_idx),
                                diff_rgb=np.abs(gt.astype(np.int16) - pred.astype(np.int16)).astype(np.uint8))
            self._plot_window_psnr_grid(self.current_window, False, plot_dir, f"window_{cur_idx}")
            self._plot_window_normal_grid(self.current_window, False, plot_dir, f"window_{cur_idx}")

        return render_pkg

    def initialize_network(self, cur_frame_idx, viewpoint, update_gaussians=False):
        if cur_frame_idx == self.dystart:
            inited = self.gaussians.create_node_from_depth(viewpoint, self.opt_params, self.sc_params)
            if not inited:
                return
        # self.gaussians.deform.deform.init(opt=self.opt_params, init_pcl=self.gaussians.get_xyz, keep_all=True, force_init=True, reset_bbox=False)
        # self.gaussians.deform.train_setting(self.sc_params)

        for mapping_iteration in tqdm(
            range(100),
            desc=self._mapper_bar_desc(f"init net[{cur_frame_idx}]"),
            position=0,
            dynamic_ncols=True,
            leave=True,
        ):
            dxyz, d_rot, d_scale, _, _ = self._get_deform_render_inputs(viewpoint, True)
            render_pkg = self._render_with_deform(
                viewpoint,
                dxyz=dxyz,
                d_rot=d_rot,
                d_scale=d_scale,
                return_normal=self.use_normal,
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
                self.config["mapping"], image, depth, viewpoint, opacity, initialization=True
            )
            if self.use_normal:
                loss_init += self.config["mapping"]["Training"]["render_normal_loss_weight"] * self.get_render_normal_loss(viewpoint.normal, normal)
                loss_init += self.config["mapping"]["Training"]["flat_loss_weight"] * self.get_flat_loss()
            # loss_init += self.gaussians.deform.reg_loss

            # scaling = self.gaussians.get_scaling
            # isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            # loss_init += 10 * isotropic_loss.mean()

            loss_init.backward()

            with torch.no_grad():
                self.gaussians.deform.optimizer.step()
                self.gaussians.deform.optimizer.zero_grad(set_to_none=True)
                if update_gaussians:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
        vis_render_process(self.gaussians, self.pipeline_params, self.background, viewpoint,
                        viewpoint.uid, self.save_dir, out_dir="init_network", mask=None, dynamic=True)
        stats_window = self.current_window if len(self.current_window) > 0 else [int(cur_frame_idx)]
        self._log_window_optimization_stats("init_network", stats_window, True)
        if self.node_traj_vis and self.dynamic_model and self.gaussians.deform_init:
            plot_dir = os.path.join(self.save_dir, "online_plots")
            self._plot_node_trajectory(
                viewpoint,
                True,
                plot_dir,
                f"init_network_{cur_frame_idx}",
                t_samp_num_override=2,
                force_t_idx=0,
                draw_history=False,
            )

    def find_closest_keyframe(self, uid):
        """仅使用时间上更早的关键帧（与 before_modify 一致）；无则更早前帧时不做光流监督。"""
        keys = [key for key in self.viewpoints if key < uid]
        if not keys:
            return None
        return max(keys)



    def map(self,stream,idx1, current_window, prune=False, iters=1, dynamic_network=False, dynamic_render=False, rm_initdy=False):
        if len(current_window) == 0:
            return
        key_opt = []
        if len(current_window) > 3:
            key_opt = self.viewpoints[current_window[0]].keyframe_selection_overlap(
                stream, self.viewpoints, self.viewpoints[current_window[2]].uid)
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

        iter_range = range(iters)
        if prune:
            iterator = iter_range
        else:
            iterator = tqdm(
                iter_range,
                desc=self._mapper_bar_desc(f"map [{idx1}]"),
                position=0,
                dynamic_ncols=True,
                leave=True,
            )
        for i in iterator:
            if i>100:
                self.iteration_count += 1

            self.last_sent += 1
            dygs_scaling = 0
            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []
            loss_network = 0
            keyframes_opt = []
            progress = 0.0 if iters <= 1 else float(i) / float(iters - 1)
            # Dynamic:Static expected ratio annealing:
            # start 2:1 (p=2/3) -> middle 1:1 (p=1/2) -> end 1:2 (p=1/3)
            if progress < 0.5:
                p_dynamic = (2.0 / 3.0) + ((1.0 / 2.0) - (2.0 / 3.0)) * (progress / 0.5)
            else:
                p_dynamic = (1.0 / 2.0) + ((1.0 / 3.0) - (1.0 / 2.0)) * ((progress - 0.5) / 0.5)
            dynamic = bool(torch.rand(1).item() < p_dynamic)

            flow_weights = self.flow_loss_start + (self.flow_loss_end - self.flow_loss_start) * progress

            if len(current_window) == len(viewpoint_stack):
                windows = current_window
            else:
                windows = viewpoint_stack
            for cam_idx in range(len(windows )):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)

                if dynamic_network and self.gaussians.deform_init:
                    time_input = self.gaussians.deform.deform.expand_time(viewpoint.fid)

                    d_values = self.gaussians.deform.step(
                        self.gaussians.get_dygs_xyz.detach(),
                        time_input ,
                        iteration=0,
                        feature=None,
                        motion_mask=self.gaussians.motion_mask,
                        camera_center=viewpoint.camera_center,
                        time_interval=self.gaussians.time_interval
                    )

                    dxyz = d_values['d_xyz']
                    d_rot = d_values['d_rotation']
                    d_scale = d_values['d_scaling']
                    d_opac, d_color = d_values['d_opacity'], d_values["d_color"]

                elif dynamic_render and self.gaussians.deform_init:
                    with torch.no_grad():
                        time_input = self.gaussians.deform.deform.expand_time(viewpoint.fid)
                        N = time_input.shape[0]
                        ast_noise = torch.randn(1, 1, device=time_input.device).expand(N, -1) * \
                                    self.gaussians.time_interval * self.gaussians.smooth_term(
                            self.iteration_count)
                        d_values = self.gaussians.deform.step(
                            self.gaussians.get_xyz.detach(),
                            time_input + ast_noise,
                            iteration=0,
                            feature=None,
                            motion_mask=self.gaussians.motion_mask,
                            camera_center=viewpoint.camera_center,
                            time_interval=self.gaussians.time_interval,
                        )
                        dxyz = d_values['d_xyz'].detach()
                        d_rot = d_values['d_rotation'].detach()
                        d_scale = d_values['d_scaling'].detach()
                        d_opac = d_values['d_opacity'].detach() if d_values['d_opacity'] else None
                        d_color = d_values["d_color"].detach() if d_values["d_color"] else None

                else:

                    dxyz = 0
                    d_rot, d_scale, d_opac, d_color = None, 0, None, None
                dygs_scaling += d_scale
                render_pkg = render(
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
                )

                (image, viewspace_point_tensor, visibility_filter,
                radii, depth, opacity, n_touched) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                if rm_initdy:
                    with torch.no_grad():
                        mask = viewpoint.reproject_mask(stream, self.viewpoints[0])
                else:
                    mask = None

                if dynamic_network and self.gaussians.deform_init:
                    closest_keyframe = self.find_closest_keyframe(viewpoint.uid)
                    if closest_keyframe is not None:
                        flow, flow_back, mask_fwd, mask_bwd,_ = viewpoint.generate_flow(
                            viewpoint.original_image.cuda(),idx1,
                            self.viewpoints[closest_keyframe].original_image.cuda()
                        )

                        time_input = self.gaussians.deform.deform.expand_time(
                            self.viewpoints[closest_keyframe].fid)

                        d_value2 = self.gaussians.deform.step(
                            self.gaussians.get_dygs_xyz.detach(),
                            time_input ,
                            iteration=0,
                            feature=None,
                            motion_mask=self.gaussians.motion_mask,
                            camera_center=self.viewpoints[closest_keyframe].camera_center,
                            time_interval=self.gaussians.time_interval,
                        )
                        d_xyz2 = d_value2["d_xyz"]

                        render_pkg2 = render_flow(
                            pc=self.gaussians,
                            viewpoint_camera1=viewpoint,
                            viewpoint_camera2=self.viewpoints[closest_keyframe],
                            d_xyz1=dxyz,
                            d_xyz2=d_xyz2,
                            d_rotation1=d_rot,
                            d_scaling1=d_scale,
                            scale_const=None,
                        )

                        coor1to2_motion = render_pkg2["render"][:2].permute(1, 2, 0)
                        dynamic_mask = (~viewpoint.motion_mask).unsqueeze(0).permute(1, 2, 0).repeat(1, 1,
                                                                                                    2).detach()

                        loss_network += flow_weights * l1_loss(flow_back * dynamic_mask,
                                                            coor1to2_motion * dynamic_mask)

                        render_pkg_back = render_flow(
                            pc=self.gaussians,
                            viewpoint_camera1=self.viewpoints[closest_keyframe],
                            viewpoint_camera2=viewpoint,
                            d_xyz1=d_xyz2,
                            d_xyz2=dxyz,
                            d_rotation1=d_value2["d_rotation"],
                            d_scaling1=d_value2["d_scaling"],
                        )
                        coor2to1_motion = render_pkg_back["render"][:2].permute(1, 2, 0)
                        dynamic_mask = (~self.viewpoints[closest_keyframe].motion_mask).unsqueeze(0).permute(1, 2,
                                                                                                            0).repeat(
                            1, 1, 2).detach()
                        loss_network += flow_weights * l1_loss(flow * dynamic_mask,
                                                            coor2to1_motion * dynamic_mask)

                    order_mask = (viewpoint.depth > 0)
                    loss_order_depth = self.get_depth_order_loss(depth, viewpoint.depth, order_mask)
                    loss_mapping += get_loss_mapping(
                        self.config["mapping"], image, depth, viewpoint, opacity,
                        rm_dynamic=not (dynamic_network or dynamic_render),
                        dynamic=dynamic,
                    ) + self.depth_order_loss_weight * loss_order_depth
                    loss_mapping += self.normal_loss_weight * self.get_loss_normal(depth, viewpoint)

                else:
                    loss_mapping += get_loss_mapping(
                        self.config["mapping"], image, depth, viewpoint, opacity,
                        rm_dynamic=not (dynamic_network or dynamic_render),
                    )

                if dynamic_network and self.gaussians.deform_init:

                    loss_network += 1e-3 * self.gaussians.deform.deform.arap_loss(
                        t=viewpoint.fid,
                        delta_t=delta * self.gaussians.time_interval,
                        t_samp_num=4,
                    )
                    loss_network+=1e-5* self.gaussians.deform.deform.acc_loss(
                        t=viewpoint.fid,
                        delta_t=5* self.gaussians.time_interval,
                    )

                    loss_network += 1e-3 * self.gaussians.deform.deform.elastic_loss(
                        t=viewpoint.fid,
                        delta_t=5*self.gaussians.time_interval ,
                    )

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                if dynamic_network and self.gaussians.deform_init:
                    time_input = self.gaussians.deform.deform.expand_time(viewpoint.fid)
                    N = time_input.shape[0]

                    d_values = self.gaussians.deform.step(
                        self.gaussians.get_dygs_xyz.detach(),
                        time_input,
                        iteration=0,
                        feature=None,
                        motion_mask=self.gaussians.motion_mask,
                        camera_center=viewpoint.camera_center,
                        time_interval=self.gaussians.time_interval,
                    )
                    dxyz = d_values['d_xyz']
                    d_rot = d_values['d_rotation']
                    d_scale = d_values['d_scaling']
                    d_opac = d_values['d_opacity']
                    d_color = d_values["d_color"]
                elif dynamic_render and self.gaussians.deform_init:
                    with torch.no_grad():
                        time_input = self.gaussians.deform.deform.expand_time(viewpoint.fid)
                        N = time_input.shape[0]
                        ast_noise = torch.randn(1, 1, device=time_input.device).expand(N, -1) * \
                                    self.gaussians.time_interval * self.gaussians.smooth_term(self.iteration_count)
                        d_values = self.gaussians.deform.step(
                            self.gaussians.get_xyz.detach(),
                            time_input + ast_noise,
                            motion_mask=self.gaussians.motion_mask,
                            camera_center=viewpoint.camera_center,
                            time_interval=self.gaussians.time_interval,
                        )
                        d_scale = d_values['d_scaling'].detach()
                        d_opac = d_values['d_opacity'].detach() if d_values['d_opacity'] else None
                        d_color = d_values["d_color"].detach() if d_values["d_color"] else None
                else:
                    dxyz = 0
                    d_rot, d_scale, d_opac, d_color = None, 0, None, None

                dygs_scaling += d_scale

                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background,
                    dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot, do=d_opac, dc=d_color,
                )
                (image, viewspace_point_tensor, visibility_filter,
                radii, depth, opacity, n_touched) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                if rm_initdy:
                    with torch.no_grad():
                        mask = viewpoint.reproject_mask(stream, self.viewpoints[0])
                else:
                    mask = None

                if dynamic_network and self.gaussians.deform_init:
                    if dynamic or True:
                        closest_keyframe = self.find_closest_keyframe(viewpoint.uid)
                        if closest_keyframe is not None:
                            flow, flow_back, mask_fwd, mask_bwd,_ = viewpoint.generate_flow(
                                viewpoint.original_image.cuda(),idx1,
                                self.viewpoints[closest_keyframe].original_image.cuda(),
                            )

                            time_input = self.gaussians.deform.deform.expand_time(
                                self.viewpoints[closest_keyframe].fid)
                            d_value2 = self.gaussians.deform.step(
                                self.gaussians.get_dygs_xyz.detach(),
                                time_input ,
                                feature=None,
                                motion_mask=self.gaussians.motion_mask,
                                camera_center=self.viewpoints[closest_keyframe].camera_center,
                                time_interval=self.gaussians.time_interval,
                            )
                            d_xyz2 = d_value2["d_xyz"]

                            render_pkg2 = render_flow(
                                pc=self.gaussians,
                                viewpoint_camera1=viewpoint,
                                viewpoint_camera2=self.viewpoints[closest_keyframe],
                                d_xyz1=dxyz,
                                d_xyz2=d_xyz2,
                                d_rotation1=d_rot,
                                d_scaling1=d_scale,
                            )
                            coor1to2_motion = render_pkg2["render"][:2].permute(1, 2, 0)
                            dynamic_mask = (~viewpoint.motion_mask).unsqueeze(0).permute(1, 2, 0).repeat(1, 1,
                                                                                                        2).detach()

                            loss_network += flow_weights * l1_loss(flow_back * dynamic_mask,
                                                                coor1to2_motion * dynamic_mask)


                            render_pkg_back = render_flow(
                                pc=self.gaussians,
                                viewpoint_camera1=self.viewpoints[closest_keyframe],
                                viewpoint_camera2=viewpoint,
                                d_xyz1=d_xyz2,
                                d_xyz2=dxyz,
                                d_rotation1=d_value2["d_rotation"],
                                d_scaling1=d_value2["d_scaling"],
                            )
                            coor2to1_motion = render_pkg_back["render"][:2].permute(1, 2, 0)
                            dynamic_mask = (~self.viewpoints[closest_keyframe].motion_mask).unsqueeze(0).permute(1, 2,
                                                                                                                0).repeat(
                                1, 1, 2).detach()
                            loss_network += flow_weights * l1_loss(flow * dynamic_mask,
                                                                coor2to1_motion * dynamic_mask)

                        order_mask = (viewpoint.depth > 0)
                        loss_order_depth = self.get_depth_order_loss(depth, viewpoint.depth, order_mask)
                        loss_mapping += get_loss_mapping(
                            self.config["mapping"], image, depth, viewpoint, opacity,
                            rm_dynamic=not (dynamic_network or dynamic_render),
                            dynamic=dynamic,
                        ) + self.depth_order_loss_weight * loss_order_depth
                        loss_mapping += self.normal_loss_weight * self.get_loss_normal(depth, viewpoint)
                    else:
                        image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
                        gt_image = viewpoint.original_image.cuda()
                        gt_depth = torch.from_numpy(viewpoint.depth).to(device=image.device)[None]
                        depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
                        l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
                        Ll1 = l1_loss(image, gt_image)
                        loss_mapping += (1.0 - self.opt_params.lambda_dssim) * Ll1 + \
                                        self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
                        loss_mapping += 0.1 * l1_depth.mean()
                else:
                    loss_mapping += get_loss_mapping(
                        self.config["mapping"], image, depth, viewpoint, opacity,
                        rm_dynamic=not (dynamic_network or dynamic_render),
                        mask=mask,
                    )

                if dynamic_network and self.gaussians.deform_init:

                    loss_network += 1e-4 * self.gaussians.deform.deform.elastic_loss(
                        t=viewpoint.fid,
                        delta_t= 5*self.gaussians.time_interval,
                    )

                    loss_network+=1e-6* self.gaussians.deform.deform.acc_loss(
                        t=viewpoint.fid,
                        delta_t=5 * self.gaussians.time_interval,
                    )
                    loss_network += 1e-4 * self.gaussians.deform.deform.arap_loss(
                        t=viewpoint.fid,
                        delta_t=5 * self.gaussians.time_interval,
                    )

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            visible_any = torch.zeros_like(visibility_filter_acm[0], dtype=torch.bool)
            for vf in visibility_filter_acm:
                visible_any = torch.logical_or(visible_any, vf)

            dynamic_prune_gate = None
            dynamic_mask_cuda = None
            if (
                self.preserve_invisible_gaussians
                and hasattr(self.gaussians, "dygs")
                and torch.is_tensor(self.gaussians.dygs)
                and self.gaussians.dygs.numel() == visible_any.numel()
            ):
                dynamic_mask_cuda = self.gaussians.dygs.bool().to(visible_any.device)
                dynamic_prune_gate = torch.where(
                    dynamic_mask_cuda,
                    visible_any,
                    torch.ones_like(visible_any, dtype=torch.bool),
                )

            scaling = self.gaussians.get_scaling
            if self.freeze_dynamic_when_invisible and dynamic_prune_gate is not None:
                visible_scaling = scaling[dynamic_prune_gate]
            else:
                visible_scaling = scaling
            if visible_scaling.numel() > 0:
                isotropic_loss = torch.abs(
                    visible_scaling - visible_scaling.mean(dim=1).view(-1, 1)
                )
                loss_mapping += 10 * isotropic_loss.mean()

            dynamic_visible_any = True
            if (
                self.freeze_dynamic_when_invisible
                and dynamic_mask_cuda is not None
            ):
                dynamic_visible_any = bool(
                    torch.logical_and(visible_any, dynamic_mask_cuda)
                    .any()
                    .item()
                )
            loss_mapping.backward(retain_graph=True)
            gaussian_split = False
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                if prune:
                    if len(current_window) == self.window_size:
                        prune_mode = self.config["mapping"]["Training"]["prune_mode"]
                        prune_coviz = self.prune_coviz
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                        if prune_mode == "slam":
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            obs_prune = self.gaussians.n_obs <= prune_coviz
                            to_prune = torch.logical_and(obs_prune, mask)

                            # Dynamic Gaussians can temporarily leave FoV.
                            # Keep a grace period and require low opacity before pruning.
                            if (
                                to_prune is not None
                                and hasattr(self.gaussians, "dygs")
                                and torch.is_tensor(self.gaussians.dygs)
                                and self.gaussians.dygs.numel() == to_prune.numel()
                            ):
                                dynamic_mask = self.gaussians.dygs.bool().cpu()
                                newest_kf = int(sorted_window[0])
                                ages = newest_kf - self.gaussians.unique_kfIDs.long().cpu()
                                grace_mask = ages <= self.dynamic_prune_grace

                                dynamic_obs_prune = self.gaussians.n_obs <= self.dynamic_prune_coviz
                                if self.preserve_invisible_gaussians:
                                    dynamic_obs_prune = torch.logical_and(dynamic_obs_prune, self.gaussians.n_obs > 0)

                                dynamic_to_prune = dynamic_obs_prune
                                dynamic_to_prune = torch.logical_and(dynamic_to_prune, ~grace_mask)
                                dynamic_to_prune = torch.logical_and(dynamic_to_prune, dynamic_mask)

                                if self.protect_dynamic_out_of_view:
                                    static_to_prune = torch.logical_and(to_prune, ~dynamic_mask)
                                    to_prune = torch.logical_or(static_to_prune, dynamic_to_prune)
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())

                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (self.occ_aware_visibility[current_idx][
                                    ~to_prune])

                        if not self.initialized:
                            self.initialized = True
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                        self.iteration_count % self.gaussian_update_every
                        == self.gaussian_update_offset and i > 100
                )
                if rm_initdy:
                    update_gaussian = (iters - i - 10 == 0)
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                        visible_mask=dynamic_prune_gate,
                    )
                    gaussian_split = True

                if (self.iteration_count % self.gaussian_reset) == 0 and (
                        not update_gaussian
                ) and i > 100:
                    self.printer.print("Resetting the opacity of non-visible Gaussians", FontColor.MAPPER)
                    self.gaussians.reset_opacity_nonvisible(
                        visibility_filter_acm,
                        preserve_dynamic_nonvisible=(
                            self.protect_dynamic_out_of_view and self.preserve_invisible_gaussians
                        ),
                    )
                    gaussian_split = True
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if self._is_reference_viewpoint(viewpoint):
                        continue
                    update_pose(viewpoint)
                if dynamic_network and self.gaussians.deform_init:
                    if (not self.freeze_dynamic_when_invisible) or dynamic_visible_any:
                        loss_network.backward()
                        self.gaussians.deform.optimizer.step()
                    self.gaussians.deform.optimizer.zero_grad(set_to_none=True)

                if i > 100:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    self.gaussians.update_learning_rate(self.iteration_count)
                else:
                    self.gaussians.optimizer.zero_grad(set_to_none=True)

        if self.online_plotting:
            from thirdparty.gaussian_splatting.utils.image_utils import psnr
            from src.utils.eval_utils import plot_rgbd_silhouette
            cur_idx = current_window[np.array(current_window).argmax()]
            viewpoint = self.viewpoints[cur_idx]
            dxyz, d_rot, d_scale, d_opac, d_color = self._get_deform_render_inputs(
                viewpoint, dynamic_network
            )
            render_pkg = self._render_with_deform(
                viewpoint,
                dxyz=dxyz,
                d_rot=d_rot,
                d_scale=d_scale,
                d_opac=d_opac,
                d_color=d_color,
                return_normal=False,
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

            if len(self.video_idxs) > 0 and viewpoint.uid != self.video_idxs[0]:
                image = (torch.exp(viewpoint.exposure_a.detach())) * image + viewpoint.exposure_b.detach()

            image = torch.clamp(image, 0.0, 1.0)
            gt = self._chw01_to_rgb_uint8_hwc(gt_image)
            pred = self._chw01_to_rgb_uint8_hwc(image)
            mask = gt_image > 0
            psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
            diff_depth_l1, depth_l1 = self._depth_l1_diff_for_online_plot(depth, gt_depth)

            plot_dir = os.path.join(self.save_dir, "online_plots")
            plot_rgbd_silhouette(gt_image, gt_depth, image, depth, diff_depth_l1,
                                psnr_score.item(), depth_l1.item(), plot_dir=plot_dir, idx=str(cur_idx),
                                diff_rgb=np.abs(gt.astype(np.int16) - pred.astype(np.int16)).astype(np.uint8))
            self._plot_window_psnr_grid(current_window, dynamic_network, plot_dir, f"window_{cur_idx}")
            self._plot_window_normal_grid(current_window, dynamic_network, plot_dir, f"window_{cur_idx}")
            window_t_samples = max(2, len(current_window))
            self._plot_node_trajectory(
                viewpoint,
                dynamic_network,
                plot_dir,
                f"window_{cur_idx}",
                t_samp_num_override=window_t_samples,
                force_t_idx=window_t_samples - 1,
                draw_history=True,
            )

        self._log_window_optimization_stats(f"map[{idx1}]", current_window, dynamic_network)

        return gaussian_split

    # ---------------------------------------------------------------------
    # Section G: Final Refinement Stage
    # ---------------------------------------------------------------------

    def _serialize_camera_state(self, camera):
        depth = None
        if camera.depth is not None:
            depth = np.asarray(camera.depth, dtype=np.float32)

        return {
            "R": camera.R.detach().cpu(),
            "T": camera.T.detach().cpu(),
            "depth": depth,
            "cam_rot_delta": camera.cam_rot_delta.detach().cpu(),
            "cam_trans_delta": camera.cam_trans_delta.detach().cpu(),
            "exposure_a": camera.exposure_a.detach().cpu(),
            "exposure_b": camera.exposure_b.detach().cpu(),
        }

    def _apply_camera_state(self, camera, state):
        camera.update_RT(
            state["R"].to(self.device, dtype=torch.float32),
            state["T"].to(self.device, dtype=torch.float32),
        )
        if state.get("depth", None) is not None:
            camera.depth = np.asarray(state["depth"], dtype=np.float32)

        if hasattr(camera, "cam_rot_delta") and state.get("cam_rot_delta", None) is not None:
            camera.cam_rot_delta.data.copy_(state["cam_rot_delta"].to(camera.cam_rot_delta.device))
        if hasattr(camera, "cam_trans_delta") and state.get("cam_trans_delta", None) is not None:
            camera.cam_trans_delta.data.copy_(state["cam_trans_delta"].to(camera.cam_trans_delta.device))
        if hasattr(camera, "exposure_a") and state.get("exposure_a", None) is not None:
            camera.exposure_a.data.copy_(state["exposure_a"].to(camera.exposure_a.device))
        if hasattr(camera, "exposure_b") and state.get("exposure_b", None) is not None:
            camera.exposure_b.data.copy_(state["exposure_b"].to(camera.exposure_b.device))

    def _build_final_refine_optimizer(self):
        opt_params = []
        frames_to_optimize = self.config["mapping"]["Training"]["pose_window"]

        for cam_rank, cam_idx in enumerate(sorted(self.viewpoints.keys())):
            viewpoint = self.viewpoints[cam_idx]
            if (not self._is_reference_viewpoint(viewpoint)) and cam_rank < frames_to_optimize:
                opt_params.append(
                    {
                        "params": [viewpoint.cam_rot_delta],
                        "lr": self.config["mapping"]["Training"]["lr"]["cam_rot_delta"] * 0.5,
                        "name": "rot_{}".format(viewpoint.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint.cam_trans_delta],
                        "lr": self.config["mapping"]["Training"]["lr"]["cam_trans_delta"] * 0.5,
                        "name": "trans_{}".format(viewpoint.uid),
                    }
                )

            if not self._is_reference_viewpoint(viewpoint):
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

        self.keyframe_optimizers = torch.optim.Adam(opt_params) if len(opt_params) > 0 else None

    def _save_final_refine_context(self, context_path):
        context_dir = os.path.dirname(context_path)
        if context_dir:
            os.makedirs(context_dir, exist_ok=True)

        gaussian_state = {
            "xyz": self.gaussians._xyz.detach().cpu(),
            "features_dc": self.gaussians._features_dc.detach().cpu(),
            "features_rest": self.gaussians._features_rest.detach().cpu(),
            "opacity": self.gaussians._opacity.detach().cpu(),
            "scaling": self.gaussians._scaling.detach().cpu(),
            "rotation": self.gaussians._rotation.detach().cpu(),
            "max_radii2D": self.gaussians.max_radii2D.detach().cpu(),
            "unique_kfIDs": self.gaussians.unique_kfIDs.detach().cpu(),
            "n_obs": self.gaussians.n_obs.detach().cpu(),
            "dygs": self.gaussians.dygs.detach().cpu(),
            "active_sh_degree": int(self.gaussians.active_sh_degree),
            "deform_init": bool(self.gaussians.deform_init),
        }

        if torch.is_tensor(self.gaussians.xyz_gradient_accum):
            gaussian_state["xyz_gradient_accum"] = self.gaussians.xyz_gradient_accum.detach().cpu()
        if torch.is_tensor(getattr(self.gaussians, "denom", None)):
            gaussian_state["denom"] = self.gaussians.denom.detach().cpu()

        payload = {
            "version": 1,
            "iteration_count": int(self.iteration_count),
            "last_sent": int(self.last_sent),
            "keyframe_idxs": [int(v) for v in self.keyframe_idxs],
            "video_idxs": [int(v) for v in self.video_idxs],
            "current_window": [int(v) for v in self.current_window],
            "mapped_video_idxs": [int(v) for v in getattr(self, "mapped_video_idxs", [])],
            "mapped_kf_idxs": [int(v) for v in getattr(self, "mapped_kf_idxs", [])],
            "st_predicted": self.st_predicted,
            "new_scale_alignFrame0": self.new_scale_alignFrame0,
            "gaussians": gaussian_state,
            "cameras": {
                int(uid): self._serialize_camera_state(cam)
                for uid, cam in self.cameras.items()
            },
            "viewpoints": {
                int(uid): self._serialize_camera_state(cam)
                for uid, cam in self.viewpoints.items()
            },
        }

        if self.dynamic_model and self.gaussians.deform_init:
            payload["deform_state"] = {
                k: v.detach().cpu() for k, v in self.gaussians.deform.deform.state_dict().items()
            }

        torch.save(payload, context_path)
        self.printer.print(f"Saved final refine context to {context_path}", FontColor.MAPPER)

    def _load_final_refine_context(self, context_path):
        if not os.path.exists(context_path):
            self.printer.print(f"Refine context not found: {context_path}", FontColor.INFO)
            return False

        payload = torch.load(context_path, map_location="cpu")
        if not isinstance(payload, dict) or "gaussians" not in payload:
            self.printer.print(f"Invalid refine context file: {context_path}", FontColor.ERROR)
            return False

        self.iteration_count = int(payload.get("iteration_count", self.iteration_count))
        self.last_sent = int(payload.get("last_sent", self.last_sent))
        self.keyframe_idxs = [int(v) for v in payload.get("keyframe_idxs", self.keyframe_idxs)]
        self.video_idxs = [int(v) for v in payload.get("video_idxs", self.video_idxs)]
        self.current_window = [int(v) for v in payload.get("current_window", self.current_window)]
        self.mapped_video_idxs = [int(v) for v in payload.get("mapped_video_idxs", getattr(self, "mapped_video_idxs", []))]
        self.mapped_kf_idxs = [int(v) for v in payload.get("mapped_kf_idxs", getattr(self, "mapped_kf_idxs", []))]
        self.st_predicted = payload.get("st_predicted", self.st_predicted)
        self.new_scale_alignFrame0 = payload.get("new_scale_alignFrame0", self.new_scale_alignFrame0)

        gs = payload["gaussians"]
        device = self.device

        self.gaussians._xyz = nn.Parameter(gs["xyz"].to(device=device, dtype=torch.float32).requires_grad_(True))
        self.gaussians._features_dc = nn.Parameter(gs["features_dc"].to(device=device, dtype=torch.float32).requires_grad_(True))
        self.gaussians._features_rest = nn.Parameter(gs["features_rest"].to(device=device, dtype=torch.float32).requires_grad_(True))
        self.gaussians._opacity = nn.Parameter(gs["opacity"].to(device=device, dtype=torch.float32).requires_grad_(True))
        self.gaussians._scaling = nn.Parameter(gs["scaling"].to(device=device, dtype=torch.float32).requires_grad_(True))
        self.gaussians._rotation = nn.Parameter(gs["rotation"].to(device=device, dtype=torch.float32).requires_grad_(True))
        self.gaussians.max_radii2D = gs["max_radii2D"].to(device=device, dtype=torch.float32)
        self.gaussians.unique_kfIDs = gs["unique_kfIDs"].to(dtype=torch.int32, device="cpu")
        self.gaussians.n_obs = gs["n_obs"].to(dtype=torch.int32, device="cpu")
        self.gaussians.dygs = gs["dygs"].to(dtype=torch.bool, device=device)
        self.gaussians.active_sh_degree = int(gs.get("active_sh_degree", self.gaussians.active_sh_degree))
        self.gaussians.deform_init = bool(gs.get("deform_init", self.gaussians.deform_init))

        if "xyz_gradient_accum" in gs:
            self.gaussians.xyz_gradient_accum = gs["xyz_gradient_accum"].to(device=device, dtype=torch.float32)
        if "denom" in gs:
            self.gaussians.denom = gs["denom"].to(device=device, dtype=torch.float32)

        self.gaussians.training_setup(self.opt_params)

        if self.dynamic_model and self.gaussians.deform_init:
            self.gaussians.deform.train_setting(self.sc_params)
            deform_state = payload.get("deform_state", None)
            if deform_state is not None:
                self.gaussians.deform.deform.load_state_dict(deform_state, strict=False)

        for uid, state in payload.get("cameras", {}).items():
            uid = int(uid)
            if uid in self.cameras:
                self._apply_camera_state(self.cameras[uid], state)

        for uid, state in payload.get("viewpoints", {}).items():
            uid = int(uid)
            if uid in self.viewpoints:
                self._apply_camera_state(self.viewpoints[uid], state)

        self.printer.print(f"Loaded final refine context from {context_path}", FontColor.MAPPER)
        return True

    def final_refine(self, prune=False, iters=26000):
        self.printer.print("Starting final refinement", FontColor.MAPPER)

        # 导入评价指标计算函数
        from thirdparty.gaussian_splatting.utils.image_utils import psnr
        from thirdparty.gaussian_splatting.utils.loss_utils import ssim
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        # 初始化评价指标记录列表
        psnr_history = []
        ssim_history = []
        lpips_history = []
        iteration_history = []

        # 初始化LPIPS计算器
        cal_lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to("cuda")

        # 创建输出目录
        refine_output_dir = os.path.join(self.save_dir, "final_refine_visualizations")
        os.makedirs(refine_output_dir, exist_ok=True)
        depth_output_dir = os.path.join(refine_output_dir, "depth")
        normal_output_dir = os.path.join(refine_output_dir, "normal")
        flow_output_dir = os.path.join(refine_output_dir, "flow")
        os.makedirs(depth_output_dir, exist_ok=True)
        os.makedirs(normal_output_dir, exist_ok=True)
        os.makedirs(flow_output_dir, exist_ok=True)

        for keyframe_idx, frame_idx in zip(self.video_idxs, self.keyframe_idxs):

            _, _, depth_gtd, _, motion_mask, normal, mono, static_msk = self.frame_reader[frame_idx]
            depth_gt_numpy = depth_gtd.cpu().numpy()

            intrinsics = as_intrinsics_matrix(self.frame_reader.get_intrinsic()).to(self.device)

            mono_depth = load_mono_depth(frame_idx, self.save_dir).to(self.device)

            depth_temp, w2c_temp, invalid = self.get_w2c_and_depth(keyframe_idx, frame_idx, mono_depth, motion_mask,
                                                                   depth_gt_numpy, normal, mono, static_msk,
                                                                   init=False)

            w2c_old = torch.cat((self.cameras[keyframe_idx].R, self.cameras[keyframe_idx].T.unsqueeze(-1)), dim=1)
            w2c_old = torch.cat((w2c_old, torch.tensor([[0, 0, 0, 1]], device="cuda")), dim=0)

            self.cameras[keyframe_idx].update_RT(w2c_temp[:3, :3], w2c_temp[:3, 3])

            self.cameras[keyframe_idx].depth = depth_temp.cpu().numpy()

            if keyframe_idx in self.viewpoints:
                self.viewpoints[keyframe_idx].update_RT(w2c_temp[:3, :3], w2c_temp[:3, 3])

                self.viewpoints[keyframe_idx].depth = depth_temp.cpu().numpy()

            if self.move_points and self.is_kf[keyframe_idx]:

                if invalid:
                    self.update_mapping_points(keyframe_idx, w2c_temp, w2c_old, depth_temp,
                                               self.depth_dict[keyframe_idx], intrinsics, method="rigid")
                else:
                    self.update_mapping_points(keyframe_idx, w2c_temp, w2c_old, depth_temp,
                                               self.depth_dict[keyframe_idx], intrinsics)

                    self.depth_dict[keyframe_idx] = depth_temp

        random_viewpoint_stack = []

        frames_to_optimize = self.config["mapping"]["Training"]["pose_window"]

        for cam_idx, viewpoint in self.viewpoints.items():
            random_viewpoint_stack.append(viewpoint)

        for iteration in tqdm(range(iters)):
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
                # 随机选择视角进行渲染------------------------------------------------
                rand_idx = np.random.randint(0, len(random_viewpoint_stack))
                viewpoint = random_viewpoint_stack[rand_idx]
                print("self.dynamic_model", self.dynamic_model)
                print("self.gaussians.deform_init", self.gaussians.deform_init)
                if self.dynamic_model and self.gaussians.deform_init:
                    time_input = self.gaussians.deform.deform.expand_time(viewpoint.fid)
                    N = time_input.shape[0]
                    d_values = self.gaussians.deform.step(self.gaussians.get_dygs_xyz.detach(), time_input,
                                                          iteration=0, feature=None,
                                                          motion_mask=self.gaussians.motion_mask,
                                                          camera_center=viewpoint.camera_center,
                                                          time_interval=self.gaussians.time_interval)
                    dxyz = d_values['d_xyz']
                    d_rot, d_scale = d_values['d_rotation'], d_values['d_scaling']
                    d_opac, d_color = d_values['d_opacity'], d_values["d_color"]
                    closest_keyframe = self.find_closest_keyframe(viewpoint.uid)
                    if closest_keyframe is not None:
                        time_input = self.gaussians.deform.deform.expand_time(
                            self.viewpoints[closest_keyframe].fid)

                        d_value2 = self.gaussians.deform.step(
                            self.gaussians.get_dygs_xyz.detach(),
                            time_input,
                            iteration=0,
                            feature=None,
                            motion_mask=self.gaussians.motion_mask,
                            camera_center=self.viewpoints[closest_keyframe].camera_center,
                            time_interval=self.gaussians.time_interval,
                        )
                        d_xyz2 = d_value2["d_xyz"]
                        d_rot2 = d_value2['d_rotation']
                        d_scale2 = d_value2['d_scaling']
                        d_opac2 = d_value2['d_opacity'] if d_value2['d_opacity'] else None
                        d_color2 = d_value2["d_color"] if d_values["d_color"] else None
                else:
                    dxyz, d_rot, d_scale, d_opac, d_color = 0, 0, 0, None, None
                    d_xyz2, d_rot2, d_scale2, d_opac2, d_color2 = 0, 0, 0, None, None

                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz,
                    ds=d_scale, dr=d_rot, do=d_opac, dc=d_color
                )

                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
                gt_image = viewpoint.original_image.cuda()
                gt_depth = torch.from_numpy(viewpoint.depth).to(
                    dtype=torch.float32, device=image.device
                )[None]
                depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
                if self.dynamic_model:
                    Ll1 = l1_loss(image, gt_image)
                    loss += (1.0 - self.opt_params.lambda_dssim) * (
                        Ll1
                    ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
                    loss += 1e-4 * self.gaussians.deform.deform.arap_loss(t=viewpoint.fid,
                                                                          delta_t=5 * self.gaussians.time_interval,
                                                                          t_samp_num=8)
                else:
                    Ll1 = l1_loss(image, gt_image, mask=viewpoint.motion_mask)
                    loss += (1.0 - self.opt_params.lambda_dssim) * (
                        Ll1
                    ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image, mask=viewpoint.motion_mask))
                    depth_pixel_mask = viewpoint.motion_mask.view(*gt_depth.shape) * depth_pixel_mask

                l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
                loss += 0.1 * l1_depth.mean()
                loss_depth = depth_loss_dpt(depth, viewpoint.depth)
                order_mask = (viewpoint.depth > 0)
                loss_order_depth = self.get_depth_order_loss(depth, viewpoint.depth, order_mask)
                loss += 0.1 * loss_order_depth  # 1e-3 * loss_depth +
                if iteration % 400 == 0 and iteration > 7000:
                    self.visualize_mask = True
                    self.visualize_mask_combination = True
                else:
                    self.visualize_mask = False
                    self.visualize_mask_combination = False
                if iteration > 7000:
                    if closest_keyframe is not None and closest_keyframe in self.viewpoints:
                        loss += self.compute_multi_view_loss(viewpoint, render_pkg, self.gaussians,
                                                             self.pipeline_params, self.background,
                                                             d_xyz2, d_scale2, d_rot2, d_opac2,
                                                             d_color2,
                                                             self.viewpoints[closest_keyframe])

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss += 10 * isotropic_loss.mean()

            loss.backward()
            gaussian_split = False

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

                # 每100次迭代计算并记录评价指标（PSNR, SSIM, LPIPS）
                # 参考eval_utils.py，遍历所有关键帧计算平均指标
                if iteration % 1000 == 0 and iteration>0:
                    try:
                        psnr_list, ssim_list, lpips_list = [], [], []

                        # 遍历所有关键帧（参考eval_utils.py）
                        for kf_idx, video_idx in zip(self.keyframe_idxs, self.video_idxs):
                            frame = self.cameras[video_idx]

                            # 获取GT图像
                            _, gt_img, gt_depth_data, _, motion_msk, normal_data, mono_data, static_mask = \
                                self.frame_reader[kf_idx]
                            gt_img = gt_img.squeeze().cuda()
                            gt_depth=gt_depth_data.cpu().numpy()
                            # 渲染当前帧
                            if self.dynamic_model and self.gaussians.deform_init:
                                time_input = self.gaussians.deform.deform.expand_time(frame.fid)
                                d_values = self.gaussians.deform.step(
                                    self.gaussians.get_dygs_xyz.detach(), time_input,
                                    iteration=0, feature=None,
                                    motion_mask=self.gaussians.motion_mask,
                                    camera_center=frame.camera_center,
                                    time_interval=self.gaussians.time_interval
                                )
                                dxyz = d_values['d_xyz']
                                d_rot, d_scale = d_values['d_rotation'], d_values['d_scaling']
                                d_opac, d_color = d_values.get('d_opacity'), d_values.get("d_color")
                            else:
                                dxyz, d_rot, d_scale, d_opac, d_color = 0, 0, 0, None, None

                            # 渲染
                            render_pkg = render(
                                frame, self.gaussians, self.pipeline_params, self.background,
                                dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot, do=d_opac, dc=d_color
                            )

                            rendered_img = render_pkg["render"].detach()

                            # 应用曝光补偿（如果不是第一帧）
                            if video_idx > 0:
                                rendered_img = (torch.exp(
                                    frame.exposure_a.detach())) * rendered_img + frame.exposure_b.detach()
                            rendered_img = torch.clamp(rendered_img, 0.0, 1.0)

                            # 准备mask（参考eval_utils.py的逻辑）
                            mask = gt_img > 0
                            depth_mask= gt_depth>0
                            # 根据模型类型调整mask
                            if not self.gaussians.deform_init:
            #print("eval remove motion region")
                                mask = mask * motion_mask.view(*depth_mask.shape) * torch.from_numpy(depth_mask).to(device=motion_mask.device)
                                depth_mask = depth_mask * motion_mask.view(*depth_mask.shape).cpu().numpy()
                            else:
                                mask = mask * torch.from_numpy(depth_mask).to(device=motion_mask.device)


                            # 计算PSNR（使用masked区域）
                            psnr_value = psnr((rendered_img[mask]).unsqueeze(0), (gt_img[mask]).unsqueeze(0))

                            # 计算SSIM（使用全图）
                            ssim_value = ssim(rendered_img.unsqueeze(0), gt_img.unsqueeze(0))

                            # 计算LPIPS（使用全图）
                            lpips_value = cal_lpips(rendered_img.unsqueeze(0), gt_img.unsqueeze(0))

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
                        print(f"\nIteration {iteration} (avg over {len(psnr_list)} frames): "
                              f"PSNR = {avg_psnr:.2f} dB, SSIM = {avg_ssim:.4f}, LPIPS = {avg_lpips:.4f}")
                    except Exception as e:
                        print(f"\nWarning: Failed to calculate metrics at iteration {iteration}: {e}")

                # 每500次迭代保存可视化结果
                # 遍历所有关键帧保存深度图、法向量图和光流图
                if iteration % 1000 == 0 and iteration > 0:
                    try:
                        print(f"\nSaving visualizations for iteration {iteration}...")

                        # 遍历所有关键帧（与评价指标计算保持一致）
                        for kf_idx, video_idx in zip(self.keyframe_idxs, self.video_idxs):
                            frame = self.cameras[video_idx]

                            # 渲染当前帧
                            if self.dynamic_model and self.gaussians.deform_init:
                                time_input = self.gaussians.deform.deform.expand_time(frame.fid)
                                d_values = self.gaussians.deform.step(
                                    self.gaussians.get_dygs_xyz.detach(), time_input,
                                    iteration=0, feature=None,
                                    motion_mask=self.gaussians.motion_mask,
                                    camera_center=frame.camera_center,
                                    time_interval=self.gaussians.time_interval
                                )
                                dxyz = d_values['d_xyz']
                                d_rot, d_scale = d_values['d_rotation'], d_values['d_scaling']
                                d_opac, d_color = d_values.get('d_opacity'), d_values.get("d_color")
                            else:
                                dxyz, d_rot, d_scale, d_opac, d_color = 0, 0, 0, None, None

                            # 渲染
                            render_pkg = render(
                                frame, self.gaussians, self.pipeline_params, self.background,
                                dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot, do=d_opac, dc=d_color
                            )

                            rendered_depth = render_pkg["depth"]

                            # === 1. 保存深度图 ===
                            depth_vis = rendered_depth[0].detach().cpu().numpy()
                            depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
                            # 使用turbo色彩映射（与其他可视化保持一致）
                            depth_colored = plt.cm.turbo(depth_vis)[:, :, :3]
                            depth_img = (depth_colored * 255).astype(np.uint8)
                            depth_filename = f"iter_{iteration:06d}_frame_{video_idx:04d}.png"
                            cv2.imwrite(os.path.join(depth_output_dir, depth_filename),cv2.cvtColor(depth_img, cv2.COLOR_RGB2BGR))

                            # === 2. 计算并保存法向量图（使用get_surface_normalv2方法）===
                            # 先对深度进行平滑滤波
                            depth_filter = torch.nn.functional.avg_pool2d(
                                rendered_depth.unsqueeze(0), kernel_size=3, stride=1, padding=1
                            )
                            depth_filter = torch.nn.functional.avg_pool2d(
                                depth_filter, kernel_size=3, stride=1, padding=1
                            )

                            # 转换深度到xyz坐标
                            focal_x = torch.Tensor([frame.fx]).cuda()
                            focal_y = torch.Tensor([frame.fy]).cuda()
                            xyz = self.depth_to_xyz(depth_filter, focal_x, focal_y)

                            # 使用get_surface_normalv2计算法向量
                            xyz_i = xyz[0, :][None, :, :, :]
                            normal_map = self.get_surface_normalv2(xyz_i).permute((3, 2, 0, 1))

                            # 转换为可视化格式
                            normal_vis = normal_map.squeeze().detach().cpu().numpy()
                            if normal_vis.shape[0] == 3:
                                normal_vis = np.transpose(normal_vis, (1, 2, 0))  # [H, W, 3]
                            normal_vis = (normal_vis + 1) / 2  # 从[-1,1]映射到[0,1]
                            normal_img = (normal_vis * 255).astype(np.uint8)
                            normal_filename = f"iter_{iteration:06d}_frame_{video_idx:04d}.png"
                            cv2.imwrite(os.path.join(normal_output_dir, normal_filename),
                                        cv2.cvtColor(normal_img, cv2.COLOR_RGB2BGR))

                            # === 3. 如果是动态模型，计算并保存光流图 ===
                            if self.dynamic_model and self.gaussians.deform_init:
                                try:
                                    # 找到最近的关键帧来计算光流
                                    closest_keyframe = self.find_closest_keyframe(frame.uid)
                                    if closest_keyframe is not None and closest_keyframe in self.viewpoints:
                                        closest_viewpoint = self.viewpoints[closest_keyframe]

                                        # 计算当前帧的变形参数（已在前面计算过，直接使用dxyz, d_rot, d_scale）
                                        # 如果需要重新计算：
                                        time_input1 = self.gaussians.deform.deform.expand_time(frame.fid)
                                        d_values1 = self.gaussians.deform.step(
                                            self.gaussians.get_dygs_xyz.detach(), time_input1,
                                            iteration=0, feature=None,
                                            motion_mask=self.gaussians.motion_mask,
                                            camera_center=frame.camera_center,
                                            time_interval=self.gaussians.time_interval
                                        )
                                        d_xyz1 = d_values1['d_xyz']
                                        d_rot1 = d_values1['d_rotation']
                                        d_scale1 = d_values1['d_scaling']
                                        
                                        # 计算最近关键帧的变形参数
                                        time_input2 = self.gaussians.deform.deform.expand_time(closest_viewpoint.fid)
                                        d_values2 = self.gaussians.deform.step(
                                            self.gaussians.get_dygs_xyz.detach(), time_input2,
                                            iteration=0, feature=None,
                                            motion_mask=self.gaussians.motion_mask,
                                            camera_center=closest_viewpoint.camera_center,
                                            time_interval=self.gaussians.time_interval
                                        )
                                        d_xyz2 = d_values2['d_xyz']
                                        d_rot2 = d_values2['d_rotation']
                                        d_scale2 = d_values2['d_scaling']
                                        
                                        # 使用正确的参数调用render_flow
                                        flow_pkg = render_flow(
                                            pc=self.gaussians,
                                            viewpoint_camera1=frame,
                                            viewpoint_camera2=closest_viewpoint,
                                            d_xyz1=d_xyz1,
                                            d_xyz2=d_xyz2,
                                            d_rotation1=d_rot1,
                                            d_scaling1=d_scale1
                                        )

                                        if "render" in flow_pkg:
                                            flow= flow_pkg["render"].detach().cpu().numpy()  # Shape: (3,H,W)
                                            flow = np.transpose(flow, (1, 2, 0))  # 修复：修正拼写错误 transflow_data → transpose
                                            # 可视化光流（使用HSV色彩空间）
                                            flow_vis = self.visualize_flow(flow)
                                            flow_filename = f"iter_{iteration:06d}_frame_{video_idx:04d}.png"
                                            cv2.imwrite(os.path.join(flow_output_dir, flow_filename), flow_vis)
                                except Exception as e:
                                    print(
                                        f"\nWarning: Failed to save flow for frame {video_idx} at iteration {iteration}: {e}")

                        print(f"Saved visualizations for {len(self.keyframe_idxs)} frames at iteration {iteration}")
                    except Exception as e:
                        print(f"\nWarning: Failed to save visualizations at iteration {iteration}: {e}")

        if len(psnr_history) > 0 and len(ssim_history) > 0 and len(lpips_history) > 0:
            # 验证列表长度一致性
            if not (len(psnr_history) == len(ssim_history) == len(lpips_history) == len(iteration_history)):
                print(f"⚠️ Warning: Metrics history length mismatch!")
                print(f"   psnr: {len(psnr_history)}, ssim: {len(ssim_history)}, "
                    f"lpips: {len(lpips_history)}, iteration: {len(iteration_history)}")
                # 截断到最短长度
                min_len = min(len(psnr_history), len(ssim_history), len(lpips_history), len(iteration_history))
                psnr_history = psnr_history[:min_len]
                ssim_history = ssim_history[:min_len]
                lpips_history = lpips_history[:min_len]
                iteration_history = iteration_history[:min_len]
                print(f"   Truncated to {min_len} samples")
            
            try:
                fig, ax1 = plt.subplots(figsize=(14, 8))

                # 左Y轴：PSNR
                color_psnr = 'tab:blue'
                ax1.set_xlabel('Iteration', fontsize=13, fontweight='bold')
                ax1.set_ylabel('PSNR', color=color_psnr, fontsize=13, fontweight='bold')
                line1 = ax1.plot(iteration_history, psnr_history, color=color_psnr, linewidth=2.5, 
                                marker='o', markersize=5, label='PSNR', alpha=0.8)
                ax1.tick_params(axis='y', labelcolor=color_psnr, labelsize=11)
                ax1.grid(True, alpha=0.3, linestyle='--')

                # 右Y轴：SSIM 和 LPIPS
                ax2 = ax1.twinx()
                color_ssim = 'tab:green'
                color_lpips = 'tab:red'
                ax2.set_ylabel('SSIM / LPIPS', fontsize=13, fontweight='bold')
                
                line2 = ax2.plot(iteration_history, ssim_history, color=color_ssim, linewidth=2.5,
                                marker='s', markersize=5, label='SSIM', alpha=0.8)
                line3 = ax2.plot(iteration_history, lpips_history, color=color_lpips, linewidth=2.5,
                                marker='^', markersize=5, label='LPIPS', alpha=0.8)
                ax2.tick_params(axis='y', labelsize=11)

                # 合并图例
                lines = line1 + line2 + line3
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='best', fontsize=12, framealpha=0.9)

                # 设置标题
                plt.title('Metrics during Final Refinement', fontsize=15, fontweight='bold', pad=20)
                plt.tight_layout()

                # 保存综合曲线图
                metrics_plot_path = os.path.join(refine_output_dir, "metrics_curve.png")
                plt.savefig(metrics_plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                # 创建包含三个子图的图表
                fig, axes = plt.subplots(3, 1, figsize=(12, 14))
        
                # ========== 保存单独的PSNR曲线图 ==========
                plt.figure(figsize=(12, 6))
                plt.plot(iteration_history, psnr_history, 'b-', linewidth=2, marker='o', markersize=4)
                plt.xlabel('Iteration', fontsize=12)
                plt.ylabel('PSNR (dB)', fontsize=12)
                plt.title('PSNR during Final Refinement (Higher is Better)', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                psnr_plot_path = os.path.join(refine_output_dir, "psnr_curve.png")
                plt.savefig(psnr_plot_path, dpi=150, bbox_inches='tight')
                plt.close()

                # ========== 保存单独的SSIM曲线图 ==========
                plt.figure(figsize=(12, 6))
                plt.plot(iteration_history, ssim_history, 'g-', linewidth=2, marker='s', markersize=4)
                plt.xlabel('Iteration', fontsize=12)
                plt.ylabel('SSIM', fontsize=12)
                plt.title('SSIM during Final Refinement (Higher is Better)', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                ssim_plot_path = os.path.join(refine_output_dir, "ssim_curve.png")
                plt.savefig(ssim_plot_path, dpi=150, bbox_inches='tight')
                plt.close()

                # ========== 保存单独的LPIPS曲线图 ==========
                plt.figure(figsize=(12, 6))
                plt.plot(iteration_history, lpips_history, 'r-', linewidth=2, marker='^', markersize=4)
                plt.xlabel('Iteration', fontsize=12)
                plt.ylabel('LPIPS', fontsize=12)
                plt.title('LPIPS during Final Refinement (Lower is Better)', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                lpips_plot_path = os.path.join(refine_output_dir, "lpips_curve.png")
                plt.savefig(lpips_plot_path, dpi=150, bbox_inches='tight')
                plt.close()
        
                # 保存评价指标数据到文本文件（添加编码）
                metrics_data_path = os.path.join(refine_output_dir, "metrics_data.txt")
                with open(metrics_data_path, 'w', encoding='utf-8') as f:
                    f.write("Iteration\tPSNR(dB)\tSSIM\tLPIPS\n")
                    for iter_num, psnr_val, ssim_val, lpips_val in zip(iteration_history, psnr_history, ssim_history,
                                                                    lpips_history):
                        f.write(f"{iter_num}\t{psnr_val:.4f}\t{ssim_val:.4f}\t{lpips_val:.4f}\n")
        
                # 保存统计摘要（添加编码）
                summary_path = os.path.join(refine_output_dir, "metrics_summary.txt")
                with open(summary_path, 'w', encoding='utf-8') as f:
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
                    f.write("LPIPS (Learned Perceptual Image Patch Similarity) - Lower is Better\n")
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
                    f"\nPSNR  - Final: {psnr_history[-1]:.2f} dB, Max: {max(psnr_history):.2f} dB, Mean: {np.mean(psnr_history):.2f} dB")
                print(
                    f"SSIM  - Final: {ssim_history[-1]:.4f}, Max: {max(ssim_history):.4f}, Mean: {np.mean(ssim_history):.4f}")
                print(
                    f"LPIPS - Final: {lpips_history[-1]:.4f}, Min: {min(lpips_history):.4f}, Mean: {np.mean(lpips_history):.4f}")
                print("=" * 60)
            
            except Exception as e:
                print(f"⚠️ Error generating metrics visualization: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️ Warning: Insufficient metrics data collected (only {len(psnr_history)} samples)")
            print(f"   Note: Metrics are collected every 100 iterations, total iterations: {iters}")

        if (
            self.node_traj_final_vis
            and self.dynamic_model
            and self.gaussians.deform_init
            and len(self.viewpoints) >= 2
        ):
            view_ids = sorted(self.viewpoints.keys())
            traj_tag = "final_refine_full_sequence"
            self._plot_window_dynamic_trajectory_2d_grid(
                view_ids, True, refine_output_dir, traj_tag, allow_without_online_vis=True
            )
            self._plot_window_dynamic_trajectory_3d(
                view_ids, True, refine_output_dir, traj_tag, allow_without_online_vis=True
            )
            if self.node_traj_gif:
                valid_indices = [int(k) for k in view_ids if int(k) in self.viewpoints]
                valid_indices = sorted(valid_indices)
                view_seq = [self.viewpoints[k] for k in valid_indices]
                payload = self._compute_dynamic_tracks_for_views(view_seq)
                if payload is not None:
                    node_plot_dir = os.path.join(refine_output_dir, "node_trajectory")
                    self._save_node_traj_gif_2d_grid(
                        payload, node_plot_dir, traj_tag, self.node_traj_gif_fps
                    )
                    self._save_node_traj_gif_3d(
                        payload, node_plot_dir, traj_tag, self.node_traj_gif_fps
                    )

        self.printer.print("Final refinement done", FontColor.MAPPER)

    def visualize_flow(self, flow):
        flow_uv = flow[..., :2].copy() * 5.0  # 放大5倍使弱运动可见

        # --- 运动方向与速度的HSV编码 ---
        magnitude = np.linalg.norm(flow_uv, axis=-1)  # 速度幅值
        angle = np.arctan2(flow_uv[..., 1], flow_uv[..., 0])  # 运动方向 [-π, π]

        # 构建HSV图像
        hsv = np.zeros((flow_uv.shape[0], flow_uv.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = (angle + np.pi) / (2 * np.pi) * 180  # 色相：方向[0°, 180°]
        hsv[..., 1] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # 饱和度：速度
        hsv[..., 2] = 255  # 亮度：固定最大值

        # 转换并保存为BGR（修复：补全字符串和变量名）
        flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return flow_img

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
        rgb_boundary_threshold = self.config["mapping"]["Training"]["rgb_boundary_threshold"]
        self.mapped_video_idxs.append(cur_frame_idx)
        self.mapped_kf_idxs.append(idx)
        #monodepth=load_mono_depth(idx, self.save_dir).to(self.device)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        with torch.no_grad():
            output_dir = os.path.join("output", "sitting2_xyz_image")
            os.makedirs(output_dir, exist_ok=True)
            depth1 = gt_img.permute(1, 2, 0).cpu().detach().numpy()
            depth_normalized = (depth1 - depth1.min()) / (
                    depth1.max() - depth1.min()) * 255
            depth_normalized = depth_normalized.astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f"image_test_{idx:04d}.png"), depth_normalized)
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
            initial_depth = initial_depth.detach().clone()  # change 0 region according to opacity rendering
            initial_depth[0][~viewpoint.motion_mask.cpu().numpy()] = 0

        return  initial_depth[0].cpu().numpy()

    def is_keyframe(
            self,
            cur_frame_idx,
            last_keyframe_idx,
            cur_frame_visibility_filter,
            occ_aware_visibility,
    ):
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

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
            self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
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

    # ---------------------------------------------------------------------
    # Section H: End-to-End Runtime Pipeline
    # ---------------------------------------------------------------------

    def run(self,stream:BaseDataset):
        """
        Trigger mapping process, get estimated pose and depth from tracking process,
        send continue signal to tracking process when the mapping of the current frame finishes.
        """
        config = self.config
        self.stream=stream
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
        # Define first frame pose
        _, color, _, first_frame_c2w,_,normal,mono,static_msk = self.frame_reader[0]
        intrinsics = as_intrinsics_matrix(self.frame_reader.get_intrinsic()).to(self.device)

        # Create dictionary which stores the depth maps from the previous iteration
        # This depth is used during map deformation if we have missing pixels
        self.depth_dict = dict()
        # global camera dictionary - updated during mapping.
        self.cameras = dict()
        self.depth_dict = dict()

        while (1):
            frame_info = self.pipe.recv()
            idx = frame_info['timestamp']  # frame index
            video_idx = frame_info['video_idx']  # keyframe index
            is_finished = frame_info['end']

            if self.verbose:
                # self.printer.print(f"\nMapping Frame {idx} ...", FontColor.MAPPER)
                pass

            if is_finished:
                self._plot_node_trajectory_full_sequence()
                print("Done with Mapping and Tracking")
                break

            if self.verbose:
                print(Fore.GREEN)
                # print("Mapping Frame ", idx)
                print(Style.RESET_ALL)

            self.keyframe_idxs.append(idx)
            self.video_idxs.append(video_idx)

            _, color, depth_gt, c2w_gt,motion_mask,normal,mono,static_msk = self.frame_reader[idx]
            mono_depth = load_mono_depth(idx, self.save_dir).to(self.device)

            color = color.to(self.device)
            c2w_gt = c2w_gt.to(self.device)
            depth_gt_numpy = depth_gt.numpy()


            depth, w2c, invalid = self.get_w2c_and_depth(video_idx, idx, mono_depth, motion_mask,depth_gt_numpy,normal,mono,static_msk,init=False)

            w2c_gt = torch.linalg.inv(c2w_gt)
            if invalid:
                print("WARNING: Too few valid pixels from droid depth")
                # online glorieslam pose and depth
                data = {"gt_color": color.squeeze(), "glorie_depth": depth.cpu().numpy(), "glorie_pose": w2c, \
                        "gt_pose": w2c_gt, "idx": video_idx}
                self.is_kf[video_idx] = False
                viewpoint = Camera.init_from_dataset(
                    self.frame_reader,video_idx,idx, data, projection_matrix
                )
                # update the estimated pose to be the glorie pose
                viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
                viewpoint.compute_grad_mask(self.config)
                # Dictionary of Camera objects at the frame index
                # self.cameras contains all cameras.
                self.cameras[video_idx] = viewpoint
                self.pipe.send("continue")
                continue  # too few valid pixels from droid depth

            #w2c_gt = torch.linalg.inv(c2w_gt)
            self.gt_w2c_all_frames.append(w2c_gt)

            # online glorieslam pose and depth
            data = {"gt_color": color.squeeze(), "glorie_depth": depth.cpu().numpy(), "glorie_pose":  w2c, \
                    "gt_pose": w2c_gt, "idx": video_idx}

            viewpoint = Camera.init_from_dataset(
                self.frame_reader,video_idx,idx, data, projection_matrix
            )
            # update the estimated pose to be the glorie pose
            viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

            viewpoint.compute_grad_mask(self.config)
            # Dictionary of Camera objects at the frame index
            # self.cameras contains all cameras.
            self.cameras[video_idx] = viewpoint

            if self.dynamic_model:
                self.gaussians.deform.deform.reg_loss = 0.

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
                depth = self.add_new_keyframe(video_idx, idx
                        )
                self.add_next_kf(
                    video_idx, idx,viewpoint, depth_map=depth, init=True,
                )
                self.initialize_map(video_idx, idx,viewpoint)
                # if self.dynamic_model and self.dystart == 11:
                #     print("dynamic0")
                #if self.dynamic_model and self.dystart ==video_idx:
                self.initialize_network(video_idx, viewpoint)
                init = False
                self.pipe.send("continue")
                continue

            dxyz, d_rot, d_scale = 0, 0, 0

            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot,
            )

            self.median_depth = get_median_depth(render_pkg["depth"], render_pkg["opacity"])

            # keyframe selection
            last_keyframe_idx = self.current_window[0]
            if len(self.keyframe_idxs) >= 4:
                last_idx=self.keyframe_idxs[-2]
            else:
                last_idx = 0
            curr_visibility = (render_pkg["n_touched"] > 0).long()

            create_kf = self.is_keyframe(
                video_idx,
                last_keyframe_idx,
                curr_visibility,
                self.occ_aware_visibility,
            ) or  (idx - last_idx) >=2
            if len(self.current_window) < self.window_size:
                # When we have not filled up the keyframe window size
                # we rely on just the covisibility thresholding, not the
                # translation thresholds.
                union = torch.logical_or(
                    curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                ).count_nonzero()
                intersection = torch.logical_and(
                    curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                ).count_nonzero()
                point_ratio = intersection / union

                create_kf = (
                        point_ratio < self.config["mapping"]["Training"]["kf_overlap"] or (idx - last_keyframe_idx) >=2
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

            for keyframe_idx, frame_idx in zip(self.video_idxs, self.keyframe_idxs):
                # need to update depth_dict even if the last idx since this is important
                # for the first deformation of the keyframe
                _, _, depth_gtd, _,motion_mask,normal,mono,static_msk = self.frame_reader[frame_idx]
                depth_gt_numpy = depth_gtd.cpu().numpy()
                mono_depth = load_mono_depth(frame_idx, self.save_dir).to(self.device)
                depth_gtd= depth_gtd.to(mono_depth.device)

                depth_temp, w2c_temp, invalid = self.get_w2c_and_depth(keyframe_idx, frame_idx, mono_depth, motion_mask,
                                                                    depth_gt_numpy,normal,mono, static_msk,init=False)
                if keyframe_idx not in self.depth_dict and self.is_kf[keyframe_idx]:
                    self.depth_dict[keyframe_idx] = depth_temp

                # No need to move the latest pose and depth
                if frame_idx != last_idx:
                    # Update tracking parameters
                    w2c_old = torch.cat((self.cameras[keyframe_idx].R, self.cameras[keyframe_idx].T.unsqueeze(-1)),
                                        dim=1)
                    w2c_old = torch.cat((w2c_old, torch.tensor([[0, 0, 0, 1]], device="cuda")), dim=0)
                    self.cameras[keyframe_idx].update_RT(w2c_temp[:3, :3], w2c_temp[:3, 3])
                    # Update depth for viewpoint
                    self.cameras[keyframe_idx].depth = depth_temp.cpu().numpy()

                    if keyframe_idx in self.viewpoints:
                        # Update tracking parameters
                        self.viewpoints[keyframe_idx].update_RT(w2c_temp[:3, :3], w2c_temp[:3, 3])
                        # Update depth for viewpoint
                        self.viewpoints[keyframe_idx].depth = depth_temp.cpu().numpy()

                    # Update mapping parameters
                    if self.move_points and self.is_kf[keyframe_idx]:
                        if invalid:
                            # if the frame was invalid, we don't update the depth old and just do a rigid correction for this frame
                            self.update_mapping_points(keyframe_idx, w2c_temp, w2c_old, depth_temp,
                                                    self.depth_dict[keyframe_idx], intrinsics, method="rigid")
                        else:
                            self.update_mapping_points(keyframe_idx, w2c_temp, w2c_old, depth_temp,
                                                    self.depth_dict[keyframe_idx], intrinsics)
                            self.depth_dict[
                                keyframe_idx] = depth_temp  # line does not matter since it is the last deformation anyway


            self.viewpoints[video_idx] = viewpoint
            depth = self.add_new_keyframe(video_idx, idx)
            self.add_next_kf(video_idx, idx,viewpoint, depth_map=depth, init=False)  # set init to True for debugging

            self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
            )

            opt_params = []
            frames_to_optimize = self.config["mapping"]["Training"]["pose_window"]
            iter_per_kf = self.mapping_itr_num
            for cam_idx in range(len(self.current_window)):
                viewpoint = self.viewpoints[self.current_window[cam_idx]]
                if self._is_reference_viewpoint(viewpoint):
                    # Do not add anchor frame pose for optimization
                    continue
                #if not self.gt_camera and self.config["mapping"]["BA"]:
                if cam_idx < frames_to_optimize:
                    opt_params.append(
                        {
                            "params": [viewpoint.cam_rot_delta],
                            "lr": self.config["mapping"]["Training"]["lr"]["cam_rot_delta"]
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

                if not self._is_reference_viewpoint(viewpoint):
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

            dynamic_enabled = self.config["mapping"]["model_params"].get("dynamic_model", False)

            self.map(stream=stream,
                    idx1=idx,
                    current_window=self.current_window,
                    iters=400,
                    dynamic_network=dynamic_enabled,
                    prune=False
                    )
            #self.map_static(current_window=self.current_window, iters=50,idx1=idx)  # 静态地图优化
            self.map(stream=stream,
                    idx1=idx,
                    current_window=self.current_window,
                    prune=True,
                    dynamic_network=dynamic_enabled
                    )

            del render_pkg, depth, w2c, depth_temp, w2c_temp

            torch.cuda.empty_cache()
            self.cleanup(video_idx)  # 常规清理
            self.pipe.send("continue")

    def cleanup(self, cur_frame_idx):

        if cur_frame_idx % 1 == 0:
            torch.cuda.empty_cache()
    def lncc(self,ref, nea):
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
        ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
        nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
        ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
        nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
        ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

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
        mask = (ncc < 0.9)
        return ncc, mask
    def patch_warp(self,H, uv):
        B, P = uv.shape[:2]
        H = H.view(B, 3, 3)
        ones = torch.ones((B, P, 1), device=uv.device)
        homo_uv = torch.cat((uv, ones), dim=-1)

        grid_tmp = torch.einsum("bik,bpk->bpi", H, homo_uv)
        grid_tmp = grid_tmp.reshape(B, P, 3)
        grid = grid_tmp[..., :2] / (grid_tmp[..., 2:] + 1e-10)
        return grid
    def patch_offsets(self,h_patch_size, device):
        offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
        return torch.stack(torch.meshgrid(offsets, offsets, indexing='xy')[::-1], dim=-1).view(1, -1, 2)

    # ---------------------------------------------------------------------
    # Section I: Multi-View Geometric/NCC Consistency Loss
    # ---------------------------------------------------------------------

    def compute_multi_view_loss(self, viewpoint_cam, render_pkg, gaussians, pipe, bg, dxyz, d_scale, d_rot, d_opac,
                                d_color,
                                nearest_cam):
        """
        计算多视角几何一致性和 NCC 损失
        """
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
            ix, iy = torch.meshgrid(
                torch.arange(W), torch.arange(H), indexing='xy')
            pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg["depth"].device)

            # 渲染最近视角
            nearest_render_pkg = render(nearest_cam, gaussians, pipe, bg, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot,
                                        do=d_opac, dc=d_color)

            # 计算3D点投影
            pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg["depth"])
            pts_in_nearest_cam = pts @ nearest_cam.R_gt + nearest_cam.T_gt
            map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg["depth"],
                                                                    pts_in_nearest_cam)

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
                [pts_in_view_cam[:, 0] * viewpoint_cam.fx / pts_in_view_cam[:, 2] + viewpoint_cam.cx,
                pts_in_view_cam[:, 1] * viewpoint_cam.fy / pts_in_view_cam[:, 2] + viewpoint_cam.cy], -1).float()

            # 计算像素噪声和权重
            pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
            d_mask_before_motion = d_mask & (pixel_noise < pixel_noise_th)  # 保存应用motion_mask之前的d_mask
            d_mask = d_mask_before_motion.clone()

            # 将d_mask与motion_mask相乘（motion_mask=True表示静态区域）
            # 直接在一维空间操作，避免reshape开销

            # 与当前视角的静态掩码相乘
            current_motion_mask = None
            if has_current_motion_mask:
                current_motion_mask = viewpoint_cam.motion_mask.reshape(-1).to(d_mask.device)
                d_mask = d_mask & current_motion_mask

            # 与临近视角的静态掩码相乘
            nearest_motion_mask = None
            if has_nearest_motion_mask:
                nearest_motion_mask = nearest_cam.motion_mask.reshape(-1).to(d_mask.device)
                d_mask = d_mask & nearest_motion_mask

            # 可视化 d_mask 和 motion_mask 的结合效果
            if hasattr(self, 'visualize_mask_combination') and self.visualize_mask_combination:
                self._visualize_mask_combination(
                    d_mask_before_motion, current_motion_mask, nearest_motion_mask, d_mask,
                    H, W, viewpoint_cam, render_pkg, render_pkg["render"]
                )

            weights = (1.0 / torch.exp(pixel_noise)).detach()
            weights[~d_mask] = 0

            # 可视化 d_mask（可选，设置 visualize_mask=True 启用）
            if hasattr(self, 'visualize_mask') and self.visualize_mask:
                self._visualize_d_mask(d_mask, pixel_noise, H, W, viewpoint_cam, render_pkg)

            # 几何损失（使用 torch.any 避免 GPU-CPU 同步）
            if torch.any(d_mask):
                # 如果不是虚拟相机，计算 NCC 损失
                if use_virtul_cam is False:
                    ncc_loss = self._compute_ncc_loss(
                        viewpoint_cam, nearest_cam, render_pkg, gt_image_gray, pixel_noise,
                        d_mask, weights, pixels, patch_size, sample_num, total_patch_size,
                        ncc_weight
                    )
                    total_loss += ncc_loss

            return total_loss

        except Exception as e:
            print(f"Error in compute_multi_view_loss: {e}")
            return 0.0

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
        axes[0, 0].set_title('渲染图像')
        axes[0, 0].axis('off')

        # 2. 渲染的深度图
        depth_vis = axes[0, 1].imshow(rendered_depth, cmap='turbo')
        axes[0, 1].set_title('渲染深度图')
        axes[0, 1].axis('off')
        plt.colorbar(depth_vis, ax=axes[0, 1])

        # 3. d_mask 掩码（黑白）
        axes[0, 2].imshow(d_mask_img, cmap='gray')
        axes[0, 2].set_title(f'd_mask (有效点: {d_mask.sum().item()}/{d_mask.numel()})')
        axes[0, 2].axis('off')

        # 4. d_mask 叠加在RGB图像上（绿色表示有效区域）
        overlay = rendered_image.copy()
        mask_colored = np.zeros_like(overlay)
        mask_colored[:, :, 1] = d_mask_img / 255.0  # 绿色通道
        axes[1, 0].imshow(overlay * 0.5 + mask_colored * 0.5)
        axes[1, 0].set_title('d_mask 叠加（绿色=有效）')
        axes[1, 0].axis('off')

        # 5. 像素重投影误差热力图
        noise_vis = axes[1, 1].imshow(pixel_noise_np, cmap='hot', vmin=0, vmax=5)
        axes[1, 1].set_title('像素重投影误差')
        axes[1, 1].axis('off')
        plt.colorbar(noise_vis, ax=axes[1, 1])

        # 6. 误差分布直方图 - 修复这里！
        d_mask_flat = d_mask_np.reshape(-1).astype(bool)  # 展平并转换为布尔数组
        valid_noise = pixel_noise_flat[d_mask_flat]
        invalid_noise = pixel_noise_flat[~d_mask_flat]

        axes[1, 2].hist(valid_noise, bins=50, alpha=0.5, label='有效点', color='green')
        axes[1, 2].hist(invalid_noise, bins=50, alpha=0.5, label='无效点', color='red')
        axes[1, 2].set_xlabel('重投影误差 (像素)')
        axes[1, 2].set_ylabel('频数')
        axes[1, 2].set_title('误差分布')
        axes[1, 2].legend()
        axes[1, 2].set_xlim(0, 10)

        plt.tight_layout()

        # 保存图像
        frame_id = viewpoint_cam.uid if hasattr(viewpoint_cam, 'uid') else 'unknown'
        save_path = os.path.join(vis_dir, f"d_mask_frame_{frame_id}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[可视化] d_mask 已保存到: {save_path}")
        print(
            f"[统计] 有效点数: {d_mask.sum().item()} / {d_mask.numel()} ({100 * d_mask.sum().item() / d_mask.numel():.2f}%)")
        print(f"[统计] 平均重投影误差 (有效): {valid_noise.mean():.3f} 像素")
        if len(invalid_noise) > 0:
            print(f"[统计] 平均重投影误差 (无效): {invalid_noise.mean():.3f} 像素")
    
    def _visualize_mask_combination(self, d_mask_before, current_motion_mask, nearest_motion_mask, 
                                    d_mask_after, H, W, viewpoint_cam, render_pkg, gt_image):
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
        num_rows = 3 if (current_motion_mask is not None or nearest_motion_mask is not None) else 2
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
        
        # 第一行：原始图像和 d_mask
        axes[0, 0].imshow(gt_image_np)
        axes[0, 0].set_title('真实图像')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(d_mask_before_np, cmap='gray')
        valid_before = d_mask_before.sum().item()
        axes[0, 1].set_title(f'd_mask (应用motion_mask前)\n有效点: {valid_before}/{d_mask_before.numel()} ({100*valid_before/d_mask_before.numel():.1f}%)')
        axes[0, 1].axis('off')
        
        # d_mask 叠加在真实图像上
        overlay_before = gt_image_np.copy()
        mask_overlay = np.zeros_like(overlay_before)
        mask_overlay[:, :, 1] = d_mask_before_np  # 绿色通道
        axes[0, 2].imshow(overlay_before * 0.6 + mask_overlay * 0.4)
        axes[0, 2].set_title('d_mask 叠加（绿色=有效）')
        axes[0, 2].axis('off')
        
        # 第二行：motion_mask
        row_idx = 1
        if current_motion_mask is not None:
            current_motion_np = current_motion_mask.reshape(H, W).detach().cpu().numpy()
            axes[row_idx, 0].imshow(current_motion_np, cmap='RdYlGn')  # 红色=动态，绿色=静态
            static_current = current_motion_mask.sum().item()
            axes[row_idx, 0].set_title(f'当前视角 motion_mask\n静态点: {static_current}/{current_motion_mask.numel()} ({100*static_current/current_motion_mask.numel():.1f}%)')
            axes[row_idx, 0].axis('off')
        else:
            axes[row_idx, 0].text(0.5, 0.5, 'No current\nmotion_mask', ha='center', va='center', fontsize=14)
            axes[row_idx, 0].axis('off')
        
        if nearest_motion_mask is not None:
            nearest_motion_np = nearest_motion_mask.reshape(H, W).detach().cpu().numpy()
            axes[row_idx, 1].imshow(nearest_motion_np, cmap='RdYlGn')  # 红色=动态，绿色=静态
            static_nearest = nearest_motion_mask.sum().item()
            axes[row_idx, 1].set_title(f'临近视角 motion_mask\n静态点: {static_nearest}/{nearest_motion_mask.numel()} ({100*static_nearest/nearest_motion_mask.numel():.1f}%)')
            axes[row_idx, 1].axis('off')
        else:
            axes[row_idx, 1].text(0.5, 0.5, 'No nearest\nmotion_mask', ha='center', va='center', fontsize=14)
            axes[row_idx, 1].axis('off')
        
        # 组合的 motion_mask（如果两者都存在）
        if current_motion_mask is not None and nearest_motion_mask is not None:
            combined_motion = (current_motion_mask & nearest_motion_mask).reshape(H, W).detach().cpu().numpy()
            axes[row_idx, 2].imshow(combined_motion, cmap='RdYlGn')
            static_combined = (current_motion_mask & nearest_motion_mask).sum().item()
            axes[row_idx, 2].set_title(f'组合 motion_mask (交集)\n静态点: {static_combined}/{current_motion_mask.numel()} ({100*static_combined/current_motion_mask.numel():.1f}%)')
            axes[row_idx, 2].axis('off')
        elif current_motion_mask is not None:
            axes[row_idx, 2].imshow(current_motion_np, cmap='RdYlGn')
            axes[row_idx, 2].set_title('组合 motion_mask\n(只有当前视角)')
            axes[row_idx, 2].axis('off')
        elif nearest_motion_mask is not None:
            axes[row_idx, 2].imshow(nearest_motion_np, cmap='RdYlGn')
            axes[row_idx, 2].set_title('组合 motion_mask\n(只有临近视角)')
            axes[row_idx, 2].axis('off')
        else:
            axes[row_idx, 2].text(0.5, 0.5, 'No motion_mask', ha='center', va='center', fontsize=14)
            axes[row_idx, 2].axis('off')
        
        # 第三行：最终结果和对比
        row_idx = 2
        axes[row_idx, 0].imshow(d_mask_after_np, cmap='gray')
        valid_after = d_mask_after.sum().item()
        axes[row_idx, 0].set_title(f'd_mask (应用motion_mask后)\n有效点: {valid_after}/{d_mask_after.numel()} ({100*valid_after/d_mask_after.numel():.1f}%)')
        axes[row_idx, 0].axis('off')
        
        # 最终 d_mask 叠加在真实图像上
        overlay_after = gt_image_np.copy()
        mask_overlay_after = np.zeros_like(overlay_after)
        mask_overlay_after[:, :, 1] = d_mask_after_np  # 绿色通道
        axes[row_idx, 1].imshow(overlay_after * 0.6 + mask_overlay_after * 0.4)
        axes[row_idx, 1].set_title('最终 d_mask 叠加')
        axes[row_idx, 1].axis('off')
        
        # 差异图：显示被 motion_mask 过滤掉的点
        diff_mask = d_mask_before_np.astype(float) - d_mask_after_np.astype(float)
        diff_img = axes[row_idx, 2].imshow(diff_mask, cmap='Reds', vmin=0, vmax=1)
        filtered_points = (d_mask_before & ~d_mask_after).sum().item()
        axes[row_idx, 2].set_title(f'被 motion_mask 过滤的点\n过滤点数: {filtered_points} ({100*filtered_points/d_mask_before.numel():.1f}%)')
        axes[row_idx, 2].axis('off')
        plt.colorbar(diff_img, ax=axes[row_idx, 2])
        
        plt.tight_layout()
        
        # 保存图像
        frame_id = viewpoint_cam.uid if hasattr(viewpoint_cam, 'uid') else 'unknown'
        save_path = os.path.join(vis_dir, f"mask_combination_frame_{frame_id}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    
    def _compute_ncc_loss(self, viewpoint_cam, nearest_cam, render_pkg, gt_image_gray,pixel_noise,
                        d_mask, weights, pixels, patch_size, sample_num, total_patch_size,
                        ncc_weight):
        """
        计算 NCC 损失（内部辅助函数）
        """
        try:
            # 初始化损失为0（而不是None）
            loss = 0.0
            geo_loss = 0.03 * ((weights * pixel_noise)[d_mask]).mean()
            loss += geo_loss
            with torch.no_grad():
                ## 采样掩码（纯GPU操作，避免同步）
                d_mask = d_mask.reshape(-1)
                valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                num_valid = valid_indices.shape[0]  # 使用 shape[0] 而不是 sum()，避免同步
                
                if num_valid > sample_num:
                    # 使用 torch.randperm 在GPU上随机采样
                    rand_indices = torch.randperm(num_valid, device=d_mask.device)[:sample_num]
                    valid_indices = valid_indices[rand_indices]

                weights_m = weights.reshape(-1)[valid_indices]

                ## 采样参考帧图像块
                pixels = pixels.reshape(-1, 2)[valid_indices]
                offsets = self.patch_offsets(patch_size, pixels.device)
                ori_pixels_patch = pixels.reshape(-1, 1, 2) / 1.0 + offsets.float()

                H, W = gt_image_gray.squeeze().shape
                pixels_patch = ori_pixels_patch.clone()
                pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2),
                                            align_corners=True)
                ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)
                ref_to_neareast_r = nearest_cam.R_gt.transpose(-1, -2) @ viewpoint_cam.R_gt
                ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.T_gt + nearest_cam.T_gt
            ## 计算单应性矩阵
            # normal_mean, _ = self.depth_to_normal(viewpoint_cam, render_pkg['depth'], world_frame=False)
            normal_mean = viewpoint_cam.normal.squeeze(0).permute(1, 2, 0)
            # normal_mean, _ = self.depth_to_normal(viewpoint_cam, viewpoint_cam.depth.to("cuda").unsqueeze(0), world_frame=False)
            ref_local_n = normal_mean.permute(1, 2, 0)
            ref_local_n = ref_local_n.reshape(-1, 3)[valid_indices]
            ix, iy = torch.meshgrid(
                torch.arange(W), torch.arange(H), indexing='xy')
            rays_d = torch.stack(
                [(ix - viewpoint_cam.cx / 1) / viewpoint_cam.fx * 1,
                (iy - viewpoint_cam.cy / 1) / viewpoint_cam.fy * 1,
                torch.ones_like(ix)], -1).float().cuda()
            ref_local_d = render_pkg["depth"] / rays_d[..., 2]
            ref_local_d = ref_local_d.reshape(-1)[valid_indices]

            H_ref_to_neareast = ref_to_neareast_r[None] - \
                                torch.matmul(ref_to_neareast_t[None, :, None].expand(ref_local_d.shape[0], 3, 1),
                                            ref_local_n[:, :, None].expand(ref_local_d.shape[0], 3, 1).permute(0, 2,
                                                                                                                1)) / \
                                ref_local_d[..., None, None]
            H_ref_to_neareast = torch.matmul(
                nearest_cam.get_k(1)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
            H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(1)

            ## 计算最近帧图像块
            grid = self.patch_warp(H_ref_to_neareast.reshape(-1, 3, 3), ori_pixels_patch)
            grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
            grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
            _, nearest_image_gray = nearest_cam.get_image()
            sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
            sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)

            ## 计算 NCC 损失
            ncc, ncc_mask = self.lncc(ref_gray_val, sampled_gray_val)
            mask = ncc_mask.reshape(-1)
            ncc = ncc.reshape(-1) * weights_m
            ncc = ncc[mask].squeeze()

            # 使用 torch.any 避免同步
            if torch.any(mask):
                ncc_loss = ncc_weight * ncc.mean()
                loss += ncc_loss
                return loss
            else:
                return loss  # 只返回 geo_loss

        except Exception as e:
            print(f"Error in _compute_ncc_loss: {e}")
            return 0.0
