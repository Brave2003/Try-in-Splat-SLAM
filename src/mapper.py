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
from lietorch import SE3, SO3
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
            # print("scale: ", d_scale)
        else:
            dxyz, d_rot, d_scale = 0, 0, 0
        render_pkg = render(
            viewpoint, gaussians, pipeline_params, background
        )
        # render_pkg = render(
        #    viewpoint, gaussians, pipeline_params, background, mask=mask, dynamic=False
        # )
        viz_im = torch.clip(render_pkg["render"].permute(1, 2, 0).detach().cpu(), 0, 1)
        # viz_depth = render_pkg['depth'][0, :, :].unsqueeze(0).detach().cpu()

        h, w, _ = viz_im.shape
        fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
        cax = ax.imshow(viz_im)
        ax.axis('off')
        # 去除空白区域
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        # 保存彩色图
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
        self.dystart=self.config["mapping"]["Training"]["dystart"] if "dystart" in self.config["mapping"]["Training"].keys() else 11
        self.video: DepthVideo = slam.video
        self.monocular=not self.initialized
        model_params = munchify(self.config["mapping"]["model_params"])
        opt_params = munchify(self.config["mapping"]["opt_params"])
        pipeline_params = munchify(self.config["mapping"]["pipeline_params"])
        self.use_spherical_harmonics = self.config["mapping"]["Training"]["spherical_harmonics"]
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )
        self.dynamic_model = self.config["mapping"]["model_params"]["dynamic_model"]
        parser = ArgumentParser(description="Training script parameters")
        hp = ModelHiddenParams(parser)

        hp = merge_hparams(hp, self.config["mapping"])
        self.sc_params=hp
        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config,args=hp,init_deform=self.config["mapping"]["model_params"]["dynamic_model"])
        self.gaussians.init_lr(6.0)
        self.st_predicted = {}
        self.list = []
        self.first_d=[]
        self.new_scale_alignFrame0 = dict()
        static_msk=np.ones( (384, 512), dtype=bool)
        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.longer=self.config["mapping"]["long"]
        self.cameras_extent = 6.0
        self.dyratio=0
        self.set_hyperparams()
        self.device = torch.device(self.config['device'])
        self.static_msk = torch.from_numpy(static_msk).to(self.device)
        if self.gaussians is None:
            raise RuntimeError("高斯模型未初始化！")
        # if self.gaussians.optimizer is None:
        #     raise RuntimeError("高斯模型的优化器未初始化！请调用 training_setup()")
        self.frame_reader = get_dataset(
            self.config, device=self.device)
        if self.config["mapping"]["model_params"]["dynamic_model"]:
            self.gaussians.deform.train_setting(hp)
            self.gaussians.time_interval = 1 / len(self.frame_reader)
    def set_pipe(self, pipe):
        self.pipe = pipe

    def set_hyperparams(self):
        mapping_config = self.config["mapping"]

        self.gt_camera = mapping_config["Training"]["gt_camera"]

        self.init_itr_num = mapping_config["Training"]["init_itr_num"]
        self.init_gaussian_update = mapping_config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = mapping_config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = mapping_config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
                self.cameras_extent * mapping_config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = mapping_config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = mapping_config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = mapping_config["Training"]["gaussian_update_offset"]
        self.gaussian_th = mapping_config["Training"]["gaussian_th"]
        self.gaussian_extent = (
                self.cameras_extent * mapping_config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = mapping_config["Training"]["gaussian_reset"]
        self.size_threshold = mapping_config["Training"]["size_threshold"]
        self.window_size = mapping_config["Training"]["window_size"]

        self.save_dir = self.config['data']['output'] + '/' + self.config['scene']

        self.move_points = self.config['mapping']['move_points']
        self.online_plotting = self.config['mapping']['online_plotting']

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
            frame_idxs = self.gaussians.unique_kfIDs  # idx which anchored the set of points
            frame_mask = (frame_idxs == frame_idx)  # global variable
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

            # 所有张量生成时指定 device
            means_cam = means_cam.to(device)
            depth = depth.to(device)
            depth_old = depth_old.to(device)  # 确保此处已移动
            rescale_scale = (1 + 1 / (means_cam[:, 2]) * (depth - depth_old)).unsqueeze(-1)  # shift
            # account for 0 depth values - then just do rigid deformation
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
    def init_image_coor(self,height, width):
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
    def depth_to_xyz(self,depth, focal_x,focal_y):
        b, c, h, w = depth.shape
        u_u0, v_v0 = self.init_image_coor(h, w)
        x = u_u0 * depth / focal_x
        y = v_v0 * depth / focal_y
        z = depth
        pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1)  # [b, h, w, c]
        # print(pw.shape)
        return pw
    def optimize_st(self,depth_torch, normal_torch, self_defined_focal_x: float,self_defined_focal_y: float):
        input_depth = depth_torch.cuda()
        input_depth = input_depth.unsqueeze(0).unsqueeze(0)
        gt_normal = normal_torch.cuda()
        focal_x = torch.Tensor([self_defined_focal_x, ]).cuda()
        focal_y=torch.Tensor([self_defined_focal_y, ]).cuda()
        s = nn.Parameter(torch.Tensor([1.]).cuda().requires_grad_(True))
        t = nn.Parameter(torch.Tensor([0.0]).cuda().requires_grad_(True))

        optimizer = torch.optim.Adam([
            {'params': s, 'lr': 1e-3},
            {'params': t, 'lr': 1e-3},

        ])
        last_step_loss = 100000.
        for step in range(500):
            optimizer.zero_grad()
            scaled_depth = s * input_depth + t
            # depth = s*input_depth+t
            depth_filter = nn.functional.avg_pool2d(scaled_depth, kernel_size=3, stride=1, padding=1)
            depth_filter = nn.functional.avg_pool2d(depth_filter, kernel_size=3, stride=1, padding=1)
            xyz = self.depth_to_xyz(depth_filter, focal_x,focal_y)
            xyz_i = xyz[0, :][None, :, :, :]
            pre_normal = self.get_surface_normalv2(xyz_i).permute((3, 2, 0, 1))
            similarity = torch.nn.functional.cosine_similarity(pre_normal, -gt_normal, dim=1)
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

    def get_surface_normalv2(self,xyz, patch_size=3):
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
    def obtain_allimgs_st(self,depth, normal, st_dict, video_idx,self_define_focal_x: float,self_define_focal_y: float):
        depth_np = depth.cpu().numpy()
        H, W = depth_np.shape
        depth_torch = torch.from_numpy(depth_np)  # (B, h, w)
        normal_torch = torch.from_numpy(normal)  # (B, h, w,3)
        if video_idx not in st_dict :
            s, t = self.optimize_st(depth_torch, normal_torch, self_define_focal_x,self_define_focal_y)
            st_dict[video_idx] = {"scale": s, "shift": t}
            print(".==========.", s, t)

    def align_all_frames(self,depth, video_id,st_predicted, new_scale_alignFrame0: dict, static_msk):
        # depth_dirs = glob(os.path.join(base_dir,'GeoWizardOut/depth_npy/*.npy'))
        if video_id==11 and len(self.list) == 0:
           self.list.append(depth)
        reference_depth = self.list[0] ## 5 is the reference frame 这个必须要和 align scale的时候用的一致。

        reference_depth = reference_depth.cpu().numpy()
        # masked_reference_depth = reference_depth
        reference_s = st_predicted[11]["scale"]
        if st_predicted[11]["shift"]<0:
            reference_t = -1.1*st_predicted[11]["shift"]#0.9
        else:
            reference_t = st_predicted[11]["shift"]

        scaled_refer_depth_nomsk = reference_s * reference_depth + reference_t
        h, w = scaled_refer_depth_nomsk.shape
        static_msk = static_msk.cpu().numpy()
        static_msk = np.array(Image.fromarray(static_msk).resize((w, h))) > 0

        scaled_refer_depth = scaled_refer_depth_nomsk[static_msk]
        del reference_t,reference_s

        Y = torch.from_numpy(scaled_refer_depth).unsqueeze(-1)

        depth = depth
        cur_s = st_predicted[video_id]["scale"]

        if self.longer:
           cur_t = max(st_predicted[video_id]["shift"],0)
        else:
           if st_predicted[video_id]["shift"] < 0:
              cur_t = -st_predicted[video_id]["shift"]
           else:
              cur_t = st_predicted[video_id]["shift"]
        cur_masked_depth = depth[static_msk]
        if not np.isnan(cur_s) or not np.isnan(cur_t):
            pass  ## 都不是nan
        else:
            cur_s = st_predicted["mean_s"]
            cur_t = st_predicted["mean_t"]
        scaled_cur_depth = cur_masked_depth * cur_s + cur_t
        # print()
        scaled_cur_depth=scaled_cur_depth.cpu().numpy()
        ##### Solving using
        A = torch.from_numpy(scaled_cur_depth).unsqueeze(-1)

        res = torch.linalg.lstsq(A, Y)
        if res.solution.item()==0:
            new_scale_alignFrame0[video_id] = 1.0
        else:
            new_scale_alignFrame0[video_id] = res.solution.item()

            previous_value = new_scale_alignFrame0[11]
            #limit=(0.2+previous_value)/previous_value
            if previous_value is not None  and self.longer:
                # 比较当前值与上一个值
                current_value = new_scale_alignFrame0[video_id]

                # 如果当前值不存在或当前值大于上一个值，则更新
                if current_value is None or current_value > previous_value*1.1 or current_value*1.1<previous_value :
                    new_scale_alignFrame0[video_id] = previous_value
                    print(f"新赋值后，video_id={video_id} 的值是: {new_scale_alignFrame0[video_id]}")
                else:
                    print(f"保持当前值，video_id={video_id} 的值是: {current_value}")
            else:
                print("没有找到上一个键的值")
        print("S,T,align_scale:", cur_s, cur_t, new_scale_alignFrame0[video_id])
        del cur_s,cur_t,res
        return new_scale_alignFrame0
        pass
    def export_scaled_pcd(self,st_predicted, new_scale_alignFrame0, depth, video_id,mean_st=False, ):

        # import open3d as o3d

        if mean_st:
            s = st_predicted["mean_s"]
            t = st_predicted["mean_t"]
        else:
            s = st_predicted[video_id]["scale"]

            if st_predicted[video_id]["shift"] < 0:
                t = -st_predicted[video_id]["shift"]#0.6
            else:
                t = st_predicted[video_id]["shift"]
            #t = max(st_predicted[video_id]["shift"], 0)
        depth = depth * s+t
        #depth[~static_msk] = depth[~static_msk]+t
        #depth[~static_msk]=depth[~static_msk]*3*s+6*t
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
        ### 用 Tanh 来近似这个符号函数。
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
    def get_w2c_and_depth(self, video_idx, idx, mono_depth,motion_mask, depth_gt,normal,mono, static_mask,print_info=False, init=False):
        """
        获取相机位姿和融合深度图（DROID深度与单目深度融合）

        参数:
            video_idx (int): 视频序列索引
            idx (int): 当前帧索引
            mono_depth (Tensor): 单目深度估计结果
            depth_gt (Tensor): 真实深度（如有）
            print_info (bool): 是否打印调试信息
            init (bool): 是否为初始化阶段

        返回:
            est_droid_depth (Tensor): 融合后的深度图
            w2c (Tensor): 世界坐标系到相机坐标系的变换矩阵
            invalid (bool): 当前帧是否有效
        # """

        est_droid_depth, valid_depth_mask, c2w = self.video.get_depth_and_pose(video_idx, self.device)
        # 计算世界坐标系到相机坐标系的变换矩阵（逆矩阵）
        c2w = c2w.to(self.device)
        w2c = torch.linalg.inv(c2w)
        os.makedirs('./camera_poses', exist_ok=True)

        # 2. 将w2c转换为numpy数组并保存
        w2c_np = w2c.cpu().numpy()
        np.save(f'./camera_poses/pose_v{video_idx}_f{idx}.npy', w2c_np)

        # 3. 同时保存人类可读的文本版本用于调试
        np.savetxt(f'./camera_poses/pose_v{video_idx}_f{idx}.txt', w2c_np, fmt='%.6f')
        # 调试信息：打印有效深度点统计
        if print_info:
            print(f"valid depth number: {valid_depth_mask.sum().item()}, "
                  f"valid depth ratio: {(valid_depth_mask.sum() / (valid_depth_mask.shape[0] * valid_depth_mask.shape[1])).item()}")

        # 有效性检查：当有效深度点不足时跳过当前帧
        if valid_depth_mask.sum() < 100:
            invalid = True
            print(
                f"Skip mapping frame {idx} at video idx {video_idx} because of not enough valid depth ({valid_depth_mask.sum()}).")
        else:
            # 预处理单目深度图 --------------------------------------------------------
            invalid = False
            est_droid_depth[~valid_depth_mask] = 0
            # 步骤1：异常值过滤（超过均值3倍的深度置零）
            mono_valid_mask = mono_depth < (mono_depth.mean() * 3)
            mono_depth[mono_depth > 3 * mono_depth.mean()] = 0

            # 步骤2：二值腐蚀操作（消除边缘噪声）
            from scipy.ndimage import binary_erosion
            mono_depth = mono_depth.cpu().numpy()
            binary_image = (mono_depth > 0).astype(int)
            iterations = 5  # 腐蚀迭代次数，控制平滑程度
            padded_binary_image = np.pad(binary_image, pad_width=iterations, mode='constant', constant_values=1)
            structure = np.ones((3, 3), dtype=int)  # 3x3腐蚀核

            # 执行腐蚀操作
            eroded_padded_image = binary_erosion(padded_binary_image,
                                                 structure=structure,
                                                 iterations=iterations)
            eroded_image = eroded_padded_image[iterations:-iterations, iterations:-iterations]

            # 步骤3：应用腐蚀掩码
            mono_depth[eroded_image == 0] = 0
            # 步骤4：图像修复（填充被腐蚀的孔洞）
            if (mono_depth == 0).sum() > 0:
                # 使用Navier-Stokes算法进行修复
                mono_depth = torch.from_numpy(
                    cv2.inpaint(mono_depth,
                                (mono_depth == 0).astype(np.uint8),
                                inpaintRadius=3,  # 修复半径
                                flags=cv2.INPAINT_NS)  # 算法选择
                ).to(self.device)
            else:
                mono_depth = torch.from_numpy(mono_depth).to(self.device)
            normal = normal.cpu().numpy()
            depth_gt = torch.from_numpy(depth_gt).to(self.device)
            print("monodepth before min",mono_depth.min())
            print("monodepth before max", mono_depth.max())
            if video_idx == 11 and len(self.first_d)  == 0:
                self.first_d.append(motion_mask)
            refer_mask=self.first_d[0]
            self.obtain_allimgs_st(mono_depth, normal, self.st_predicted, video_idx, 535.4, 539.2)
            mean_s = 0
            mean_t = 0
            invalid = 0

            self.st_predicted["mean_s"] = 0.8
            self.st_predicted["mean_t"] = 0.15
            self.static_msk = motion_mask*refer_mask
            self.align_all_frames(mono_depth, video_idx, self.st_predicted, self.new_scale_alignFrame0, self.static_msk)
            mono_depth = self.export_scaled_pcd(self.st_predicted, self.new_scale_alignFrame0, mono_depth, video_idx,
                                                 mean_st=False )
            mono_depth_wq = mono_depth
            torch.cuda.empty_cache()
        return mono_depth_wq, w2c, invalid

    def initialize_map(self, cur_frame_idx,idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background,
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
            loss_init = get_loss_mapping(
                self.config["mapping"], image, depth, viewpoint, opacity, initialization=True,rm_dynamic=not (self.dystart==cur_frame_idx)
            )
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

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        vis_render_process(self.gaussians, self.pipeline_params, self.background, viewpoint,
                           viewpoint.uid, self.save_dir, out_dir="map", mask=None, dynamic=False)
        self.printer.print("Initialized map", FontColor.MAPPER)

        # online plotting
        if self.online_plotting:
            from thirdparty.gaussian_splatting.utils.image_utils import psnr
            from src.utils.eval_utils import plot_rgbd_silhouette
            import cv2
            import numpy as np
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
            plot_rgbd_silhouette(gt_image, gt_depth, image, depth, diff_depth_l1,
                                 psnr_score.item(), depth_l1, plot_dir=plot_dir, idx=str(cur_idx),
                                 diff_rgb=np.abs(gt - pred))

        return render_pkg
    def find_closest_keyframe(self, uid):
        keys = [key for key in self.viewpoints if key < uid]
        if not keys:
            return None
        closest_key = max(keys)
        return closest_key




    def map(self,stream,idx1, current_window, prune=False, iters=1, dynamic_network=False, dynamic_render=False, rm_initdy=False):
        if len(current_window) == 0:
            return
        key_opt = []
        if len(current_window) > 3:
            key_opt = self.viewpoints[current_window[0]].keyframe_selection_overlap(
                stream, self.viewpoints, self.viewpoints[current_window[2]].uid)
        key_opt = current_window[:3] + key_opt
        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in key_opt]
        print(f"[Debug] current_window长度={len(current_window)}, viewpoint_stack长度={len(viewpoint_stack)}")
        random_viewpoint_stack = []  # 随机采样视角集合
        frames_to_optimize = self.config["mapping"]["Training"]["pose_window"]
        current_window_set = set(key_opt)
        print(f"[Debug] 初始窗口: {current_window}")
        print(f"[Debug] 重叠度筛选结果: {key_opt}")
        print(f"[Debug] 最终 viewpoint_stack: {[v.uid for v in viewpoint_stack]}")
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        # 动态参数初始化
        flow_weights = self.config["mapping"]["Training"]["flow_loss"]
        delta = self.config["mapping"]["Training"].get("delta", 5)

        for i in range(iters):
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
            if i < iters / 2:
                dynamic = True
                flow_weights =self.config["mapping"]["Training"]["flow_loss"]#30
            else:
                dynamic = False
                flow_weights =self.config["mapping"]["Training"]["flow_loss_fine"] if "flow_loss_fine" in self.config["mapping"][
                    "Training"] else self.config["mapping"]["Training"]["flow_loss"]#30

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
                # 执行场景渲染
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
                    loss_depth = depth_loss_dpt(depth, viewpoint.depth,order_mask)
                    loss_order_depth = self.get_depth_order_loss(depth, viewpoint.depth, order_mask)
                    loss_mapping += get_loss_mapping(
                        self.config["mapping"], image, depth, viewpoint, opacity,
                        rm_dynamic=not (dynamic_network or dynamic_render),
                        dynamic=dynamic,
                    )+0.1*loss_order_depth
                    loss_mapping += 0.1 * self.get_loss_normal(depth, viewpoint) / 10.
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
                        iteration=0,  # 当前迭代次数
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
                            #feature=None,
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

                # 渲染随机视角
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

                # 动态网络损失计算(与主循环相同逻辑)
                if dynamic_network and self.gaussians.deform_init:
                    if dynamic or True:  # 调试用强制启用
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

                            # 反向光流损失
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
                        loss_order_depth=self.get_depth_order_loss(depth,viewpoint.depth,order_mask)
                        loss_mapping += get_loss_mapping(
                            self.config["mapping"], image, depth, viewpoint, opacity,
                            rm_dynamic=not (dynamic_network or dynamic_render),
                            dynamic=dynamic,
                        )+0.1*loss_order_depth
                        loss_mapping += 0.1 * self.get_loss_normal(depth, viewpoint) / 10.
                    else:  # 静态优化分支(当前未启用)
                        image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
                        gt_image = viewpoint.original_image.cuda()
                        gt_depth = torch.from_numpy(viewpoint.depth).to(device=image.device)[None]
                        depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
                        l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
                        Ll1 = l1_loss(image, gt_image)
                        loss_mapping += (1.0 - self.opt_params.lambda_dssim) * Ll1 + \
                                        self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
                        loss_mapping += 0.1 * l1_depth.mean()
                else:  # 非动态网络损失
                    loss_mapping += get_loss_mapping(
                        self.config["mapping"], image, depth, viewpoint, opacity,
                        rm_dynamic=not (dynamic_network or dynamic_render),
                        mask=mask,
                    )

                if dynamic_network and self.gaussians.deform_init:
                    # 弹性形变约束（抑制过度拉伸）
                    loss_network += 1e-4 * self.gaussians.deform.deform.elastic_loss(
                        t=viewpoint.fid,  # 当前时间戳
                        delta_t= 5*self.gaussians.time_interval,  # 时间差分范围
                    )

                    loss_network+=1e-6* self.gaussians.deform.deform.acc_loss(
                        t=viewpoint.fid,
                        delta_t=5 * self.gaussians.time_interval,
                    )
                    loss_network += 1e-4 * self.gaussians.deform.deform.arap_loss(
                        t=viewpoint.fid,
                        delta_t=5 * self.gaussians.time_interval,
                    )


                # 累积渲染数据
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)


            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward(retain_graph=True)
            gaussian_split = False
            # Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # compute the visibility of the gaussians
                # Only prune on the last iteration and when we have a full window
                if prune:
                    if len(current_window) == self.window_size:
                        prune_mode = self.config["mapping"]["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":  # SLAM模式剪枝
                            sorted_window = sorted(current_window, reverse=True)  # 时间倒序排列
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]  # 选择最近三个关键帧创建的高斯
                            if not self.initialized:  # 初始化阶段放宽条件
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(  # 联合可见性和时间条件
                                self.gaussians.n_obs <= prune_coviz,
                                mask,
                            )
                            # to_prune = torch.logical_or(torch.logical_and(self.gaussians.dygs==True, (self.gaussians.n_obs >= 1).cuda()), to_prune.cuda())  ##
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())  # 调用剪枝函数
                            # 更新可见性统计
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
                if rm_initdy:  # 初始化动态移除模式特殊处理
                    update_gaussian = (iters - i - 10 == 0)  # 最后10次迭代前强制更新
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True  # not used it seems

                ## Opacity reset
                # self.iteration_count is a global parameter. We use gaussian reset
                # every 2001 iterations meaning if we use 60 per mapping frame
                # and there are 160 keyframes in the sequence, we do resetting
                # 4 times. Using more mapping iterations leads to more resetting
                # which can prune away more gaussians.
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                        not update_gaussian
                ) and i > 100: #and idx1 < 100:
                    print("iteration",self.iteration_count)
                    print("update_gaussian",update_gaussian)
                    print("idx",idx1)
                    print("i",i)
                    self.printer.print("Resetting the opacity of non-visible Gaussians", FontColor.MAPPER)
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True
                # comment for debugging
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
                if dynamic_network and self.gaussians.deform_init :

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

            if viewpoint.uid != self.video_idxs[0]:  # first mapping frame is reference for exposure
                image = (torch.exp(viewpoint.exposure_a.detach())) * image + viewpoint.exposure_b.detach()

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
            plot_rgbd_silhouette(gt_image, gt_depth, image, depth, diff_depth_l1,
                                 psnr_score.item(), depth_l1, plot_dir=plot_dir, idx=str(cur_idx),
                                 diff_rgb=np.abs(gt - pred))

        return gaussian_split

    def final_refine(self, prune=False,iters=26000):
        self.printer.print("Starting final refinement", FontColor.MAPPER)


        for keyframe_idx, frame_idx in zip(self.video_idxs, self.keyframe_idxs):

            _, _, depth_gtd, _, motion_mask,normal ,mono,static_msk = self.frame_reader[frame_idx]
            depth_gt_numpy = depth_gtd.cpu().numpy()


            intrinsics = as_intrinsics_matrix(self.frame_reader.get_intrinsic()).to(self.device)


            mono_depth = load_mono_depth(frame_idx, self.save_dir).to(self.device)

            depth_temp, w2c_temp, invalid = self.get_w2c_and_depth(keyframe_idx, frame_idx, mono_depth ,motion_mask,depth_gt_numpy,normal,mono,static_msk,
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

                else:
                    dxyz, d_rot, d_scale, d_opac, d_color = 0, 0, 0, None, None

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
                loss +=  0.1 * loss_order_depth#1e-3 * loss_depth +
                if iteration < 7000:
                    loss += 0.1 * self.get_loss_normal(depth, viewpoint)
                else:
                    loss += 0.1 * self.get_loss_normal(depth, viewpoint) / 2


                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss+= 10 * isotropic_loss.mean()

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

        self.printer.print("Final refinement done", FontColor.MAPPER)

    def get_loss_normal(self,depth_mean, viewpoint):
        prior_normal = viewpoint.normal.cuda()
        prior_normal = prior_normal.reshape(3, *depth_mean.shape[-2:]).permute(1, 2, 0)
        prior_normal_normalized = torch.nn.functional.normalize(prior_normal, dim=-1)

        normal_mean, _ = self.depth_to_normal(viewpoint, depth_mean, world_frame=False)
        normal_error = 1 - (prior_normal_normalized * normal_mean).sum(dim=-1)
        normal_error[prior_normal.norm(dim=-1) < 0.2] = 0
        return normal_error.mean()

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
        print("mono",self.monocular)
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
        print("init")
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
                self.printer.print(f"\nMapping Frame {idx} ...", FontColor.MAPPER)

            if is_finished:
                print("Done with Mapping and Tracking")
                break

            if self.verbose:
                print(Fore.GREEN)
                print("Mapping Frame ", idx)
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
                self.gaussians.deform.deform.reg_loss = 0.  # 重置正则化损失

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
            print("current_window",len(self.current_window))
            for cam_idx in range(len(self.current_window)):
                if self.current_window[cam_idx] == 0:
                    # Do not add GT frame pose for optimization
                    continue
                viewpoint = self.viewpoints[self.current_window[cam_idx]]
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
            print("dynamic",dynamic_enabled)

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
    def initialize_network(self, cur_frame_idx, viewpoint, update_gaussians=False):
        if cur_frame_idx == self.dystart:
            inited = self.gaussians.create_node_from_depth(viewpoint, self.opt_params, self.sc_params)
            if not inited:
                return
        # self.gaussians.deform.deform.init(opt=self.opt_params, init_pcl=self.gaussians.get_xyz, keep_all=True, force_init=True, reset_bbox=False)
        # self.gaussians.deform.train_setting(self.sc_params)

        time_input = self.gaussians.deform.deform.expand_time(viewpoint.fid)
        for mapping_iteration in range(100):
            d_values = self.gaussians.deform.step(self.gaussians.get_dygs_xyz.detach(), time_input,
                                                  iteration=0, feature=None,
                                                  motion_mask=self.gaussians.motion_mask,
                                                  camera_center=viewpoint.camera_center,
                                                  time_interval=self.gaussians.time_interval)  # , detach_node=False)
            dxyz = d_values['d_xyz']
            #print("dxyz",dxyz)
            # d_rot, d_scale = 0., 0.
            d_rot, d_scale = d_values['d_rotation'], d_values['d_scaling']
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz, ds=d_scale,dr=d_rot

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

            loss_init = get_loss_mapping(
                self.config["mapping"], image, depth, viewpoint, opacity, initialization=True
            )
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