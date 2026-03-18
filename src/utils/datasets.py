# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import copy
from thirdparty.gaussian_splatting.utils.graphics_utils import focal2fov
import torchvision.transforms as transforms
from ultralytics import YOLO


def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header["dataWindow"]
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header["channels"]:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if "Y" not in header["channels"] else channelData["Y"]

    return Y


# 单目深度缓存，避免同一帧在 mapping 多轮/多关键帧循环中重复读盘
_MONO_DEPTH_CACHE: dict = {}
_MONO_DEPTH_CACHE_MAX = 128


def load_mono_depth(idx, path):
    """加载单目深度图，带 LRU 缓存以减少重复 I/O。"""
    key = (idx, path)
    if key in _MONO_DEPTH_CACHE:
        return _MONO_DEPTH_CACHE[key]
    mono_depth_path = f"{path}/mono_priors/depths/{idx:05d}.npy"
    mono_depth = np.load(mono_depth_path)
    mono_depth_tensor = torch.from_numpy(mono_depth)
    if len(_MONO_DEPTH_CACHE) >= _MONO_DEPTH_CACHE_MAX:
        # FIFO 逐出一个
        first_key = next(iter(_MONO_DEPTH_CACHE))
        del _MONO_DEPTH_CACHE[first_key]
    _MONO_DEPTH_CACHE[key] = mono_depth_tensor
    return mono_depth_tensor


def get_dataset(cfg, device="cuda:0"):
    return dataset_dict[cfg["dataset"]](cfg, device=device)


class BaseDataset(Dataset):
    def __init__(self, cfg, device="cuda:0"):
        # 调用父类构造函数
        super(BaseDataset, self).__init__()

        # 设置数据集名称和设备（默认为cuda:0）
        self.name = cfg["dataset"]
        self.device = device

        # 获取深度图像缩放比例
        self.png_depth_scale = cfg["cam"]["png_depth_scale"]

        # 初始化一些未定义的参数
        self.n_img = -1  # 图片数量，默认为-1
        self.depth_paths = None  # 深度图像路径
        self.color_paths = None  # 彩色图像路径
        self.poses = None  # 相机姿态
        self.normal_paths = None
        self.mask_paths = None
        self.mono_paths = None
        self.static_paths = None
        self.image_timestamps = None  # 图片时间戳

        # 从配置文件中获取相机内参的高度、宽度、焦距、光心位置
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = (
            cfg["cam"]["H"],
            cfg["cam"]["W"],
            cfg["cam"]["fx"],
            cfg["cam"]["fy"],
            cfg["cam"]["cx"],
            cfg["cam"]["cy"],
        )

        # 保存原始的内参
        self.fx_orig, self.fy_orig, self.cx_orig, self.cy_orig = (
            self.fx,
            self.fy,
            self.cx,
            self.cy,
        )

        # 获取输出图像的尺寸
        self.H_out, self.W_out = cfg["cam"]["H_out"], cfg["cam"]["W_out"]

        # 获取图像边缘的尺寸（可能用于后处理或者填充）
        self.H_edge, self.W_edge = cfg["cam"]["H_edge"], cfg["cam"]["W_edge"]

        # 计算加上边缘后输出图像的尺寸
        self.H_out_with_edge, self.W_out_with_edge = (
            self.H_out + self.H_edge * 2,
            self.W_out + self.W_edge * 2,
        )

        # 将相机内参转为torch张量
        self.intrinsic = torch.as_tensor([self.fx, self.fy, self.cx, self.cy]).float()

        # 根据输出图像的尺寸调整相机内参
        self.intrinsic[0] *= self.W_out_with_edge / self.W
        self.intrinsic[1] *= self.H_out_with_edge / self.H
        self.intrinsic[2] *= self.W_out_with_edge / self.W
        self.intrinsic[3] *= self.H_out_with_edge / self.H

        # 调整光心位置，使其适应边缘
        self.intrinsic[2] -= self.W_edge
        self.intrinsic[3] -= self.H_edge
        self.yolo_model = None
        # 更新相机内参的值
        self.fx = self.intrinsic[0].item()
        self.fy = self.intrinsic[1].item()
        self.cx = self.intrinsic[2].item()
        self.cy = self.intrinsic[3].item()

        self.fovx = focal2fov(self.fx, self.W_out)
        self.fovy = focal2fov(self.fy, self.H_out)

        self.distortion = (
            np.array(cfg["cam"]["distortion"]) if "distortion" in cfg["cam"] else None
        )

        # retrieve input folder as temporary folder
        self.input_folder = os.path.join(
            cfg["data"]["dataset_root"], cfg["data"]["input_folder"]
        )
        self.dynamic_objects = 0
        self.seg_chair = True if "seg_chair" in cfg["meshing"].keys() else False
        # 前端保存的 motion_mask 图片目录，后端/数据集从此读掩码
        self.motion_mask_dir = os.path.join(
            cfg["data"]["output"], cfg["scene"], "motion_mask"
        )
        # 掩码灰度约定：True=静态。默认 False 表示磁盘上 255=动态(人)、0=静态(背景)，读入时用 (img<=127) 得到 static=True 仅背景；若你处 255=静态，则在 config 中设 motion_mask_255_is_static: true
        self.motion_mask_255_is_static = cfg.get(
            "motion_mask_255_is_static",
            cfg.get("data", {}).get("motion_mask_255_is_static", True),
        )
        # 是否使用 Depth Anything 预测深度替代真值深度（data.use_depth_anything 或顶层 use_depth_anything）
        self.use_depth_anything = cfg.get(
            "use_depth_anything", cfg.get("data", {}).get("use_depth_anything", False)
        )
        # 运行时保存的 mono 深度目录（与 load_mono_depth 一致：output/scene/mono_priors/depths）
        self.mono_priors_dir = None
        if cfg.get("data") and cfg["data"].get("output") and cfg.get("scene"):
            self.mono_priors_dir = os.path.join(
                cfg["data"]["output"], cfg["scene"], "mono_priors", "depths"
            )
        # YOLO 权重路径，默认 pretrained/yolo11l-seg.pt（与 download_pretrained.sh 一致）
        self.yolo_pretrained = cfg.get("data", {}).get(
            "yolo_pretrained", "pretrained/yolo11l-seg.pt"
        )

    # def yolo_model(self):
    #     if self._yolo_model is not None:
    #         #raise AttributeError("YOLO模型未加载，请先调用load_yolo()")
    #       return self._yolo_model

    def load_yolo(self, model_path=None):
        """正确的模型加载方法（移除@property装饰器）。未传 path 时使用 cfg 中的 yolo_pretrained（默认 pretrained/yolo11l-seg.pt）。"""
        # 检查设备可用性
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用，无法加载YOLO模型")
        if model_path is None:
            model_path = getattr(self, "yolo_pretrained", "pretrained/yolo11l-seg.pt")

        # 加载并分配设备
        self.yolo_model = YOLO(model_path).to(self.device)
        # print(f"YOLO已加载至设备：{self.yolo_model.device}")

    def __len__(self):
        return self.n_img

    def depthloader(self, index, depth_paths, depth_scale):
        if depth_paths is None:
            return None
        depth_path = depth_paths[index]
        if ".png" in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif ".exr" in depth_path:
            depth_data = readEXR_onlydepth(depth_path)
        else:
            raise TypeError(depth_path)
        depth_data = depth_data.astype(np.float32) / depth_scale

        return depth_data

    # 修改后的 datasets.py 部分代码
    # def depthloader(self,index, depth_paths, depth_scale):
    #     # 检查索引有效性
    #     if index >= len(depth_paths):
    #         raise IndexError(f"索引 {index} 越界，深度图列表长度：{len(depth_paths)}")
    #
    #     depth_path = depth_paths[index]
    #
    #     # 检查文件存在性
    #     if not os.path.exists(depth_path):
    #         raise FileNotFoundError(f"深度图文件不存在：{depth_path}")
    #
    #     # 读取文件
    #     try:
    #         if depth_path.endswith(".png"):
    #             depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    #         elif depth_path.endswith(".exr"):
    #             depth_data = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    #         else:
    #             raise ValueError(f"不支持的深度图格式：{depth_path}")
    #     except Exception as e:
    #         raise RuntimeError(f"读取深度图失败：{str(e)}")
    #
    #     # 检查数据有效性
    #     if depth_data is None:
    #         raise ValueError(f"无法解析深度图：{depth_path}")
    #     if not np.issubdtype(depth_data.dtype, np.number):
    #         raise TypeError(f"深度图数据非数值类型：{depth_path}")
    #
    #     # 检查缩放因子
    #     if depth_scale <= 0 or not isinstance(depth_scale, (int, float)):
    #         raise ValueError(f"无效的深度缩放因子：{depth_scale}")
    #
    #     # 转换和缩放
    #     depth_data = depth_data.astype(np.float32) / depth_scale
    #     return depth_data

    def get_color(self, index):
        # not used now
        color_path = self.color_paths[index]
        color_data_fullsize = cv2.imread(color_path)
        if self.distortion is not None:
            K = np.eye(3)
            K[0, 0], K[0, 2], K[1, 1], K[1, 2] = (
                self.fx_orig,
                self.cx_orig,
                self.fy_orig,
                self.cy_orig,
            )
            # undistortion is only applied on color image, not depth!
            color_data_fullsize = cv2.undistort(color_data_fullsize, K, self.distortion)

        color_data = cv2.resize(
            color_data_fullsize, (self.W_out_with_edge, self.H_out_with_edge)
        )
        color_data = (
            torch.from_numpy(color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :]
            / 255.0
        )  # bgr -> rgb, [0, 1]
        color_data = color_data.unsqueeze(dim=0)  # [1, 3, h, w]

        # crop image edge, there are invalid value on the edge of the color image
        if self.W_edge > 0:
            edge = self.W_edge
            color_data = color_data[:, :, :, edge:-edge]

        if self.H_edge > 0:
            edge = self.H_edge
            color_data = color_data[:, :, edge:-edge, :]
        return color_data

    def get_intrinsic(self):
        H_out_with_edge, W_out_with_edge = (
            self.H_out + self.H_edge * 2,
            self.W_out + self.W_edge * 2,
        )
        intrinsic = torch.as_tensor(
            [self.fx_orig, self.fy_orig, self.cx_orig, self.cy_orig]
        ).float()
        intrinsic[0] *= W_out_with_edge / self.W
        intrinsic[1] *= H_out_with_edge / self.H
        intrinsic[2] *= W_out_with_edge / self.W
        intrinsic[3] *= H_out_with_edge / self.H
        if self.W_edge > 0:
            intrinsic[2] -= self.W_edge
        if self.H_edge > 0:
            intrinsic[3] -= self.H_edge
        return intrinsic

    def depth_to_normal(
        self, depth, smooth_depth=True, use_median_filter=False, k_scale=1.0
    ):
        """从深度图估计法向量（mapper 同款）：相机反投影 + 邻域叉乘。

        先将深度反投影到相机坐标系 3D 点，再用
        normalize((Pr - Pl) x (Pd - Pu)) 计算像素法向。
        """
        dev = depth.device if isinstance(depth, torch.Tensor) else self.device
        if not isinstance(depth, torch.Tensor):
            depth = torch.from_numpy(depth).float().to(dev)
        else:
            depth = depth.to(dev)
        H, W = depth.shape[0], depth.shape[1]
        depth_4d = depth.unsqueeze(0).unsqueeze(0)

        if smooth_depth:
            if use_median_filter:
                depth_np = depth.cpu().numpy()
                depth_np = cv2.medianBlur(depth_np.astype(np.float32), 3)
                depth_4d = (
                    torch.from_numpy(depth_np).float().to(dev).unsqueeze(0).unsqueeze(0)
                )
            else:
                gaussian = (
                    torch.tensor(
                        [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                        dtype=torch.float32,
                        device=dev,
                    )
                    / 16.0
                )
                gaussian = gaussian.view(1, 1, 3, 3)
                depth_4d = F.conv2d(depth_4d, gaussian, padding=1)

        depth = depth_4d.squeeze(0).squeeze(0)

        fx = torch.tensor(self.fx, device=dev, dtype=depth.dtype)
        fy = torch.tensor(self.fy, device=dev, dtype=depth.dtype)
        cx = torch.tensor(self.cx, device=dev, dtype=depth.dtype)
        cy = torch.tensor(self.cy, device=dev, dtype=depth.dtype)

        u, v = torch.meshgrid(
            torch.arange(W, device=dev, dtype=depth.dtype),
            torch.arange(H, device=dev, dtype=depth.dtype),
            indexing="xy",
        )
        x = (u - cx) / fx * depth
        y = (v - cy) / fy * depth
        pts = torch.stack([x, y, depth * k_scale], dim=-1)

        pts_l = torch.roll(pts, shifts=1, dims=1)
        pts_r = torch.roll(pts, shifts=-1, dims=1)
        pts_u = torch.roll(pts, shifts=1, dims=0)
        pts_d = torch.roll(pts, shifts=-1, dims=0)

        normal_hwc = torch.cross(pts_r - pts_l, pts_d - pts_u, dim=-1)
        normal_hwc = F.normalize(normal_hwc, dim=-1)

        valid = depth > 1e-6
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
        normal = normal_hwc.permute(2, 0, 1)
        return normal

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        color_data_fullsize = cv2.imread(color_path)
        if self.distortion is not None:
            K = np.eye(3)
            K[0, 0], K[0, 2], K[1, 1], K[1, 2] = (
                self.fx_orig,
                self.cx_orig,
                self.fy_orig,
                self.cy_orig,
            )
            # undistortion is only applied on color image, not depth!
            color_data_fullsize = cv2.undistort(color_data_fullsize, K, self.distortion)

        outsize = (self.H_out_with_edge, self.W_out_with_edge)

        color_data = cv2.resize(
            color_data_fullsize, (self.W_out_with_edge, self.H_out_with_edge)
        )
        color_data = (
            torch.from_numpy(color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :]
            / 255.0
        )  # bgr -> rgb, [0, 1]
        # color_data = torch.from_numpy(color_data).float().permute(2, 0, 1)
        if self.yolo_model is None:
            self.load_yolo()  # 使用 self.yolo_pretrained（来自 cfg data.yolo_pretrained）
            # print("yolo true")
        color_data = color_data.unsqueeze(dim=0)  # [1, 3, h, w]
        mono_data = None
        if self.mono_paths:
            mono_data = self.mono_paths[index]
            mono_data = np.load(mono_data)
            mono_data = mono_data.astype(np.float32)
            if mono_data is not None:
                mono_data = torch.from_numpy(
                    mono_data
                ).float()  # / self.png_depth_scale
                mono_data = F.interpolate(
                    mono_data[None, None], outsize, mode="nearest"
                )[0, 0]

                if self.W_edge > 0:
                    edge = self.W_edge
                    mono_data = mono_data[:, edge:-edge]
                if self.H_edge > 0:
                    edge = self.H_edge
                    mono_data = mono_data[edge:-edge, :]
            # print("monodata",mono_data.shape)
        # 深度来源：若启用 use_depth_anything 则用 Depth Anything 预测深度，否则用真值深度
        depth_data_fullsize = None
        if getattr(self, "use_depth_anything", False):
            mono_depth_path = None
            if getattr(self, "mono_priors_dir", None) and os.path.isfile(
                os.path.join(self.mono_priors_dir, f"{index:05d}.npy")
            ):
                mono_depth_path = os.path.join(self.mono_priors_dir, f"{index:05d}.npy")
            elif self.mono_paths and index < len(self.mono_paths):
                mono_depth_path = self.mono_paths[index]
            if mono_depth_path is not None:
                depth_data_fullsize = np.load(mono_depth_path).astype(np.float32)
        if depth_data_fullsize is None:
            depth_data_fullsize = self.depthloader(
                index, self.depth_paths, self.png_depth_scale
            )
        if depth_data_fullsize is not None:
            if not isinstance(depth_data_fullsize, torch.Tensor):
                depth_data_fullsize = torch.from_numpy(depth_data_fullsize).float()
            else:
                depth_data_fullsize = depth_data_fullsize.float()
            # 真值深度在 depthloader 里已除 png_depth_scale；Depth Anything 预测深度保持原尺度（后续对齐会处理）
            depth_data = F.interpolate(
                depth_data_fullsize[None, None], outsize, mode="nearest"
            )[0, 0]

        if self.static_paths:
            static_data = self.static_paths[index]
            static_msk = None
            static_data = np.load(static_data)
            dist_flow = np.linalg.norm(static_data["flow"], ord=2, axis=-1)
            dist_flow = cv2.resize(
                dist_flow, (self.W_out_with_edge, self.H_out_with_edge)
            )
            if static_msk is None:
                print("init msk")
                static_msk = np.ones_like(dist_flow)

            static_msk = np.logical_and(static_msk, dist_flow < 0.8)
            if self.W_edge > 0:
                edge = self.W_edge
                static_msk = static_msk[:, edge:-edge]
            if self.H_edge > 0:
                edge = self.H_edge
                static_msk = static_msk[edge:-edge, :]
            print("sta", static_msk.shape)
        else:
            static_msk = None
        # crop image edge, there are invalid value on the edge of the color image
        if self.W_edge > 0:
            edge = self.W_edge
            color_data = color_data[:, :, :, edge:-edge]
            depth_data = depth_data[:, edge:-edge]
        if self.H_edge > 0:
            edge = self.H_edge
            color_data = color_data[:, :, edge:-edge, :]
            depth_data = depth_data[edge:-edge, :]

        # 法向量改为从真实深度计算，不再从 normal_paths 加载
        if self.normal_paths:
            try:
                normal_path = self.normal_paths[index]
                normal_data = np.load(normal_path)
                normal_data = cv2.cvtColor(normal_data, cv2.COLOR_BGR2RGB)
                normal_data = cv2.resize(
                    normal_data, (self.W_out_with_edge, self.H_out_with_edge)
                )
                normal_data = (
                    torch.from_numpy(normal_data).float().permute(2, 0, 1) / 255.0
                )
                normal_data = normal_data.unsqueeze(dim=0)
                if self.W_edge > 0:
                    edge = self.W_edge
                    normal_data = normal_data[:, :, :, edge:-edge]
                if self.H_edge > 0:
                    edge = self.H_edge
                    normal_data = normal_data[:, :, edge:-edge, :]
                normal_input = normal_data * 2.0 - 1.0
            except Exception as e:
                print(f"Failed to load normal map: {e}")
        if depth_data_fullsize is not None:
            normal_input = self.depth_to_normal(depth_data)
        else:
            normal_input = torch.zeros(
                3, self.H_out, self.W_out, device=self.device, dtype=torch.float32
            )

        # --- 4. 掩码：优先使用前端保存的 motion_mask 图片，否则默认全静态 ---

        # 若前端已保存该帧的 motion_mask 图片，则直接读取作为掩码（后端选择时从盘读）
        mask_from_disk = None
        if hasattr(self, "motion_mask_dir") and self.motion_mask_dir:
            path = os.path.join(self.motion_mask_dir, f"{index:06d}.png")
            if os.path.isfile(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # motion_mask 约定：True=静态，False=动态。若磁盘上 255=静态、0=动态 则 (img>127)；若 255=动态、0=静态 则取反 (img<=127)
                    if getattr(self, "motion_mask_255_is_static", False):
                        mask_from_disk = (img > 127).astype(np.float32)
                    else:
                        mask_from_disk = (img <= 127).astype(np.float32)
                    mask_from_disk = (
                        torch.from_numpy(mask_from_disk).unsqueeze(0).unsqueeze(0)
                    )
                    mask_from_disk = (
                        F.interpolate(
                            mask_from_disk,
                            size=(self.H_out, self.W_out),
                            mode="nearest",
                        )
                        .squeeze()
                        .bool()
                        .to(self.device)
                    )

        if mask_from_disk is not None:
            final_mask = mask_from_disk  # True=静态
        else:
            final_mask = torch.ones(
                (self.H_out, self.W_out), device=self.device, dtype=torch.bool
            )
        if self.poses is not None:
            pose = torch.from_numpy(
                self.poses[index]
            ).float()  # torch.from_numpy(np.linalg.inv(self.poses[0]) @ self.poses[index]).float()
        else:
            pose = None
        # print(f"Raw pose at index {index}:{self.poses[index]}")
        color_data_fullsize = cv2.cvtColor(color_data_fullsize, cv2.COLOR_BGR2RGB)
        color_data_fullsize = color_data_fullsize / 255.0
        color_data_fullsize = torch.from_numpy(color_data_fullsize)
        return (
            index,
            color_data,
            depth_data,
            pose,
            final_mask,
            normal_input,
            mono_data,
            static_msk,
        )


class Replica(BaseDataset):
    def __init__(self, cfg, device="cuda:0"):
        super(Replica, self).__init__(cfg, device)
        stride = cfg["stride"]
        max_frames = cfg["max_frames"]
        if max_frames < 0:
            max_frames = int(1e5)
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        self.n_img = len(self.color_paths)

        self.load_poses(f"{self.input_folder}/traj.txt")
        self.color_paths = self.color_paths[:max_frames][::stride]
        self.depth_paths = self.depth_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]

        self.w2c_first_pose = np.linalg.inv(self.poses[0])

        self.n_img = len(self.color_paths)

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            self.poses.append(c2w)


class ScanNet(BaseDataset):
    def __init__(self, cfg, device="cuda:0"):
        super(ScanNet, self).__init__(cfg, device)
        stride = cfg["stride"]
        max_frames = cfg["max_frames"]
        if max_frames < 0:
            max_frames = int(1e5)
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, "color", "*.jpg")),
            key=lambda x: int(os.path.basename(x)[:-4]),
        )[:max_frames][::stride]
        self.depth_paths = sorted(
            glob.glob(os.path.join(self.input_folder, "depth", "*.png")),
            key=lambda x: int(os.path.basename(x)[:-4]),
        )[:max_frames][::stride]
        self.load_poses(os.path.join(self.input_folder, "pose"))
        self.poses = self.poses[:max_frames][::stride]

        self.n_img = len(self.color_paths)
        print("INFO: {} images got!".format(self.n_img))

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(
            glob.glob(os.path.join(path, "*.txt")),
            key=lambda x: int(os.path.basename(x)[:-4]),
        )
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(" ")))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            self.poses.append(c2w)


class CoFusion(BaseDataset):
    def __init__(self, cfg, device="cuda:0"):
        super(CoFusion, self).__init__(cfg, device)
        self.dynamic = cfg.get("dynamic_model", False)
        self.seg_chair = cfg.get("seg_chair", False)
        self.seg_car = cfg.get("seg_car", False)
        self.seg_ball = cfg.get("seg_ball", False)
        (
            self.color_paths,
            self.depth_paths,
            self.poses,
            self.mask_paths,
            self.normal_paths,
            self.mono_paths,
            self.static_paths,
        ) = self.load_cofusion(self.input_folder)

        stride = cfg["stride"]
        max_frames = cfg["max_frames"]
        if max_frames < 0:
            max_frames = int(1e5)

        self.color_paths = self.color_paths[:max_frames][::stride]
        self.depth_paths = self.depth_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]
        self.mask_paths = self.mask_paths[:max_frames][::stride]
        self.normal_paths = self.normal_paths[:max_frames][::stride]
        self.mono_paths = self.mono_paths[:max_frames][::stride]
        self.static_paths = self.static_paths[:max_frames][::stride]
        self.w2c_first_pose = np.linalg.inv(self.poses[0])

        self.n_img = len(self.color_paths)

    def extract_number(self, f):
        """Extract numeric parts from a filename - 增强版"""
        import re

        filename = os.path.basename(f)

        # 1. 尝试匹配带前缀的格式：Color00001, Colour0001, depth0001, mask0001
        pattern_with_prefix = re.match(r"^[A-Za-z_]+0*(\d+)\.", filename, re.IGNORECASE)
        if pattern_with_prefix:
            return int(pattern_with_prefix.group(1))

        # 2. 尝试匹配纯数字格式：0161.png, 0001.png
        pattern_pure_digits = re.match(r"^0*(\d+)\.", filename)
        if pattern_pure_digits:
            return int(pattern_pure_digits.group(1))

        # 3. 提取所有数字，取最长的连续数字串
        all_numbers = re.findall(r"\d+", filename)
        if all_numbers:
            # 找到最长的数字串（通常这就是序号）
            longest = max(all_numbers, key=len)
            return int(longest)

        # 4. 都没有找到，返回文件名
        print(f"Warning: Could not extract number from {filename}")
        return 0

    def parse_list(self, filepath, skiprows=0):
        """read list data"""
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def load_cofusion(self, datapath):
        """read video data in cofusion format"""
        # Load color images
        color_paths = sorted(glob.glob(os.path.join(datapath, "colour", "*.png")))
        # Load depth images
        depth_paths = sorted(
            glob.glob(os.path.join(datapath, "depth_noise", "*.exr"))
            + glob.glob(os.path.join(datapath, "depth", "*.png"))
        )

        # Load mask images (optional) - check multiple possible directory names
        mask_paths = []
        mask_dirs = ["mask_color", "mask_colour", "render_mask", "mask", "dynamic_mask"]
        for mask_dir in mask_dirs:
            mask_dir_path = os.path.join(datapath, mask_dir)
            if os.path.isdir(mask_dir_path):
                mask_paths = sorted(
                    glob.glob(os.path.join(datapath, mask_dir, "*.png")),
                    key=self.extract_number,
                )
                if mask_paths:  # 确保找到了文件
                    print(
                        f"✅ Using mask images from: {mask_dir}/ ({len(mask_paths)} files)"
                    )
                    break
        if not mask_paths:
            print(f"⚠️  No mask directory found (searched: {', '.join(mask_dirs)})")

        # If seg_teddy or seg_clock is enabled, use extract_number for sorting
        if self.seg_car or self.seg_ball:
            color_paths = sorted(color_paths, key=self.extract_number)
            depth_paths = sorted(depth_paths, key=self.extract_number)
            if mask_paths:
                mask_paths = sorted(mask_paths, key=self.extract_number)

        num_imgs = len(color_paths)

        # Load poses from trajectory file
        poses = []
        trajectory_path = os.path.join(datapath, "trajectories")
        if os.path.isfile(os.path.join(trajectory_path, "gt-cam-0.txt")):
            pose_list = os.path.join(trajectory_path, "gt-cam-0.txt")
            pose_data = self.parse_list(pose_list)
            pose_vecs = pose_data[:, 0:].astype(np.float64)

            for i in range(num_imgs):
                quat = pose_vecs[i][4:]  # quaternion
                trans = pose_vecs[i][1:4]  # translation

                # Convert quaternion to rotation matrix
                import trimesh

                T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
                T[:3, 3] = trans

                # Store the inverted pose (world to camera)
                poses.append(np.linalg.inv(T))
        else:
            # No pose file found, use identity poses
            poses = [np.eye(4) for _ in range(num_imgs)]
            print("Warning: No trajectory file found, using identity poses")

        # Handle optional data paths
        # Normal paths (not available in CoFusion)
        normal_paths = []
        if os.path.isdir(os.path.join(datapath, "normal")):
            normal_paths = sorted(
                glob.glob(os.path.join(datapath, "normal", "*.npy")),
                key=self.extract_number,
            )
            # print("Using normal maps (normal)")
        else:
            normal_paths = [None] * num_imgs

        # Mono depth paths (not available in CoFusion)
        mono_paths = []
        if os.path.isdir(os.path.join(datapath, "depth_npy")):
            mono_paths = sorted(
                glob.glob(os.path.join(datapath, "depth_npy", "*.npy")),
                key=self.extract_number,
            )
            # print("Using mono depth (depth_npy)")
        else:
            mono_paths = [None] * num_imgs

        # Static/flow paths (not available in CoFusion)
        static_paths = []
        if os.path.isdir(os.path.join(datapath, "flow_i1")):
            static_paths = sorted(
                glob.glob(os.path.join(datapath, "flow_i1", "*.npz")),
                key=self.extract_number,
            )
            print("Using flow maps")
        else:
            static_paths = [None] * num_imgs

        # Ensure mask_paths has the same length
        if not mask_paths:
            mask_paths = [None] * num_imgs

        # Ensure all lists have the same length
        min_length = min(len(color_paths), len(depth_paths), len(poses))
        # min_length = 636
        color_paths = color_paths[:min_length]
        depth_paths = depth_paths[:min_length]
        poses = poses[:min_length]
        mask_paths = (
            mask_paths[:min_length]
            if mask_paths[0] is not None
            else [None] * min_length
        )
        normal_paths = (
            normal_paths[:min_length] if normal_paths else [None] * min_length
        )
        mono_paths = mono_paths[:min_length] if mono_paths else [None] * min_length
        static_paths = (
            static_paths[:min_length] if static_paths else [None] * min_length
        )

        print(f"Loaded {min_length} frames from CoFusion dataset")

        return (
            color_paths,
            depth_paths,
            poses,
            mask_paths,
            normal_paths,
            mono_paths,
            static_paths,
        )

    def pose_matrix_from_quaternion(self, pvec):
        """convert quaternion to 4x4 pose matrix"""
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose


class TUM_RGBD(BaseDataset):
    def __init__(self, cfg, device="cuda:0"):
        super(TUM_RGBD, self).__init__(cfg, device)
        (
            self.color_paths,
            self.depth_paths,
            self.poses,
            self.mask_paths,
            self.normal_paths,
            self.mono_paths,
            self.static_paths,
        ) = self.loadtum(self.input_folder, frame_rate=32)
        stride = cfg["stride"]
        max_frames = cfg["max_frames"]
        if max_frames < 0:
            max_frames = int(1e5)

        self.color_paths = self.color_paths[:max_frames][::stride]
        self.depth_paths = self.depth_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]
        self.mask_paths = self.mask_paths[:max_frames][::stride]
        self.normal_paths = self.normal_paths[:max_frames][::stride]
        self.mono_paths = self.mono_paths[:max_frames][::stride]
        self.w2c_first_pose = np.linalg.inv(self.poses[0])

        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        """read list data"""
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """pair images, depths, and poses"""
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

    def extract_number(self, file_path):

        import re

        basename = os.path.basename(file_path)
        match = re.search(r"^(\d+\.\d+)", basename)
        if match:
            return float(match.group(1))
        return 0.0

    def loadtum(self, datapath, frame_rate=-1):
        """read video data in tum-rgbd format"""
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")
        mask_paths = []
        if os.path.isdir(os.path.join(datapath, "render_mask")):
            self.mask_path = sorted(
                glob.glob(os.path.join(self.input_folder, "render_mask", "*.png")),
                key=self.extract_number,
            )
            # print(self.mask_path)
            # print("Using render mask images")
        else:
            self.mask_path = None
        normal_paths = []
        if os.path.isdir(os.path.join(datapath, "normal")):
            self.normal_path = sorted(
                glob.glob(
                    os.path.join(self.input_folder, "normal", "normal_npz", "*.npy")
                ),
                key=self.extract_number,
            )
            # print("Using normal maps (normal)")
        else:
            self.normal_path = None
        mono_paths = []
        if os.path.isdir(os.path.join(datapath, "depth_npy")):
            self.mono_path = sorted(
                glob.glob(os.path.join(self.input_folder, "depth_npy", "*.npy")),
                key=self.extract_number,
            )
            # print("Using mono depth (depth_npy)")
        else:
            self.mono_path = None
        static_paths = []
        if os.path.isdir(os.path.join(datapath, "flow_RAFT1")):
            self.static_path = sorted(
                glob.glob(os.path.join(self.input_folder, "flow_RAFT1", "*.npz")),
                key=self.extract_number,
            )
            print("Using flow RAFT static (flow_RAFT1)")
        else:
            self.static_path = None
        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            # timestamp tx ty tz qx qy qz qw
            if self.mask_path is not None:
                mask_file = self.mask_path[i]
                mask_paths += [mask_file]

            if self.normal_path is not None:
                normal_paths += [self.normal_path[i]]
            if self.mono_path is not None:
                mono_paths += [self.mono_path[i]]
            if self.static_path is not None:
                static_paths += [self.static_path[i]]
            # c2w = pose_vecs[k]  # 提取相机到世界坐标系的位姿向量
            # c2w = torch.from_numpy(c2w).float()  # 转为PyTorch张量

            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose @ c2w

            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            poses += [c2w]

        return images, depths, poses, mask_paths, normal_paths, mono_paths, static_paths

    def pose_matrix_from_quaternion(self, pvec):
        """convert 4x4 pose matrix to (t, q)"""
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose


dataset_dict = {
    "replica": Replica,
    "scannet": ScanNet,
    "tumrgbd": TUM_RGBD,
    "CoFusion": CoFusion,
}
