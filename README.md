<!-- PROJECT LOGO -->
  <h1 align="center">Anonymous github</h1>


<p align="center">
This is not an officially endorsed Google product.
</p>


<p align="center">
    <img src="./media/teaser.png" alt="teaser_image" width="100%">
</p>

<p align="center">
<strong>Anonymous github
</p>

<p align="center">
    <img src="./media/framework.png" alt="framework" width="100%">
</p>
<p align="center">
<strong>Anonymous Architecture</strong>.
</p>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#run">Run</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#contact">Contact</a>
    </li>
  </ol>
</details>


## Installation
1. Clone the repo using the `--recursive` flag 
```bash
git clone --recursive https://github.com/enterfutures/Anonymous.git
cd Anonymous
```
2. Creating a new conda environment. 
```bash
conda create --name Anonymous python=3.10
conda activate Anonymous
```
3. Install CUDA 11.7 using conda and torch 1.13.1
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
> Now make sure that "which python" points to the correct python
executable. Also test that cuda is available
python -c "import torch; print(torch.cuda.is_available())"

4. Update depth rendering hyperparameter in thirparty library
  > By default, the gaussian rasterizer does not render gaussians that are closer than 0.2 (meters) in front of the camera. In our monocular setting, where the global scale is ambiguous, this can lead to issues during rendering. Therefore, we adjust this threshold to 0.001 instead of 0.2. Change the value at [this](https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose/blob/43e21bff91cd24986ee3dd52fe0bb06952e50ec7/cuda_rasterizer/auxiliary.h#L154) line, i.e. it should read
```bash
if (p_view.z <= 0.001f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
```
5. Install the remaining dependencies.
```bash
python -m pip install -e thirdparty/lietorch/
python -m pip install -e thirdparty/diff-gaussian-rasterization-w-pose/
python -m pip install -e thirdparty/simple-knn/
python -m pip install -e thirdparty/evaluate_3d_reconstruction_lib/
```

6. Check installation.
```bash
python -c "import torch; import lietorch; import simple_knn; import
diff_gaussian_rasterization; print(torch.cuda.is_available())"
```

7. Now install the droid backends and the other requirements
```bash
python -m pip install -e .
python -m pip install -r requirements.txt
python -m pip install pytorch-lightning==1.9 --no-deps
pip install lightning-utilities==0.4.2
pip install ultralytics
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

8. Download pretrained models (统一放在 `pretrained/` 下，推荐一键脚本).
```bash
# 在项目根目录执行，将下载 droid / depth_anything_v2_vitl / yolo11l-seg / sam2.1_hiera_base_plus / raft-things（光流）
bash scripts/download_pretrained.sh
```
<details>
  <summary>[Directory structure of pretrained (click to expand)]</summary>
  
```bash
  .
  └── pretrained
        ├── .gitkeep
        ├── droid.pth
        ├── depth_anything_v2_vitl.pth
        ├── yolo11l-seg.pt
        ├── sam2.1_hiera_base_plus.pt
        └── raft-things.pth
```
</details>

## Data Download
### 预训练模型（已由上面脚本覆盖）
若需手动下载：depth_anything_v2_vitl、yolo11l-seg、sam2.1_hiera_base_plus、droid、raft-things 均放入 `pretrained/`，参见 `scripts/download_pretrained.sh` 中的 URL。
If it is not in the class segmented by the YOLO model, then optical flow can be cloned below to generate a mask.
```bash
git clone https://github.com/zhengqili/Neural-Scene-Flow-Fields.git
```
Use YOLO and optical flow to generate a segmentation mask, then input it into SAM to obtain the final segmentation.
```bash
git clone https://github.com/facebookresearch/sam2.git
```
### 光流模型
`raft-things.pth` 已由 `scripts/download_pretrained.sh` 下载到 `pretrained/`。若需 GMA 可选：`gma-things.pth` 见 https://github.com/zacjiang/GMA/blob/main/checkpoints/gma-things.pth，放入 `pretrained/` 后可在代码中改为使用 `pretrained/gma-things.pth`。
### TUM-RGBD
```bash
bash scripts/download_tum.sh
```
Please change the `input_folder` path in the scene specific config files to point to where the data is stored.
### preprocessing

Enter this task and use it to generate normal vectors, then place the generated normal vectors into the corresponding datasets folder https://github.com/fuxiao0719/GeoWizard then run bash rename picture name
```bash
python new.py   source_dir  target_dir

```
## Run
For running Anonymous, each scene has a config folder, where the `input_folder`,`output` paths need to be specified. Below, we show some example run commands for one scene from each dataset.

### TUM-RGBD
To run Anonymous on the `freiburg3_sitting_rpy` scene, run the following command. 
```bash
python run.py configs/TUM_RGBD/rgbd_dataset_freiburg3_sitting_rpy.yaml
```
After reconstruction, the trajectory error will be evaluated automatically.


## Run tracking without mapping
Our Anonymous pipeline uses two processes, one for tracking and one for mapping, and it is possible to run tracking only without mapping/rendering. Add `--only_tracking` in each of the above commands.
```bash
python run.py configs/TUM_RGBD/rgbd_dataset_freiburg3_sitting_rpy.yaml --only_tracking
```

## Acknowledgement
Our codebase is partially based on [splat-SLAM](https://github.com/google-research/Splat-SLAM), [GO-SLAM](https://github.com/youmi-zym/GO-SLAM), [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) and [MonoGS](https://github.com/muskie82/MonoGS). We thank the authors for making these codebases publicly available. Our work would not have been possible without your great efforts!

## Reproducibility
There may be minor differences between the released codebase and the results reported in the paper. Further, we note that the GPU hardware has an influence, despite running the same seed and conda environment.

## 提交到自己的 GitHub 仓库
若要将本仓库整理后推送到你自己的 GitHub，请参阅 [GITHUB_PUSH.md](GITHUB_PUSH.md)。



