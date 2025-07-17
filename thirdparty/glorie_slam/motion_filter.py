# Copyright 2024 The GlORIE-SLAM Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import lietorch

import thirdparty.glorie_slam.geom.projective_ops as pops
from thirdparty.glorie_slam.modules.droid_net import CorrBlock
from src.mono_estimators import get_mono_depth_estimator,predict_mono_depth
from src.utils.datasets import load_mono_depth

class MotionFilter:
    """ This class is used to filter incoming frames and extract features 
        mainly inherited from DROID-SLAM
    """

    def __init__(self, net, video, cfg, thresh=2.5, device="cuda:0"):
        self.cfg = cfg
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        self.omni_dep=None
        self.dynamic_later=cfg["tracking"]["motion_filter"]["dynamic_later"]
        self.dynamic_thresh=cfg["tracking"]["motion_filter"]["dynamic_thresh"]
        self.dynamic_frame=cfg["tracking"]["motion_filter"]["dynamic_frame"]
        if cfg["mono_prior"]["predict_online"]:
            #self.mono_depth_estimator,self.mono_depth_estimator1 = get_mono_depth_estimator(cfg)
            self.mono_depth_estimator = get_mono_depth_estimator(cfg)
        self.dy=cfg["mono_prior"]["dy"]



    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)



    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth,pose,intrinsics=None,mask=None):
        """ main update operation - run on every frame in video """

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // self.video.down_scale
        wd = image.shape[-1] // self.video.down_scale

        # normalize images
        inputs = image[None, :, :].to(self.device)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)
        # extract features
        gmap = self.__feature_encoder(inputs)
        # weight_prior = 0.4  # 给 prior_depth 的权重
        # weight_mono = 0.6  # 给 mono_depth 的权重
        ### always add first frame to the depth video ###
        if self.dy:
            motion_mask = self.__get_motion_mask(image, intrinsics)
            self.video.all_motion_masks[self.k] = motion_mask
            self.k += 1
        else:
            motion_mask = torch.zeros(1, 480, 640, device=self.device).bool()
        if self.video.counter.value == 0:
            net, inp = self.__context_encoder(inputs[:,[0]])
            self.net, self.inp, self.fmap = net, inp, gmap
            print("shape",image.shape)
            if self.cfg["mono_prior"]["predict_online"]:

                mono_depth=predict_mono_depth(self.mono_depth_estimator,tstamp,depth,image,self.cfg,self.device,mask)

            else:
                mono_depth = load_mono_depth(tstamp,self.cfg)
            self.video.append(tstamp, image[0], Id, 1.0, mono_depth, intrinsics / float(self.video.down_scale), gmap, net[0,0], inp[0,0],mask,pose)

        else:                
            # index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)

            if self.dynamic_later and tstamp > self.dynamic_frame:
                self.thresh=self.dynamic_thresh
            if delta.norm(dim=-1).mean().item() > self.thresh:
                self.count = 0
                net, inp = self.__context_encoder(inputs[:,[0]])
                self.net, self.inp, self.fmap = net, inp, gmap
                if self.cfg["mono_prior"]["predict_online"]:

                    depth = depth.to(self.device)
                    mono_depth = predict_mono_depth(self.mono_depth_estimator, tstamp,depth, image, self.cfg, self.device,mask)

                else:
                    mono_depth = load_mono_depth(tstamp,self.cfg)
                self.video.append(tstamp, image[0], None, None, mono_depth, intrinsics / float(self.video.down_scale), gmap, net[0], inp[0],mask,pose)

            else:
                self.count += 1
