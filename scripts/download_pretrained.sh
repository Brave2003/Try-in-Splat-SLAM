#!/bin/bash
# 将 try 项目所需的预训练模型下载到 pretrained/ 目录
# 在 try 项目根目录下执行: bash scripts/download_pretrained.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PRETRAINED="$TRY_ROOT/pretrained"
mkdir -p "$PRETRAINED"
cd "$PRETRAINED"

download() {
    local out="$1"
    local url="$2"
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$out" "$url"
    elif command -v curl &> /dev/null; then
        curl -# -L -o "$out" "$url"
    else
        echo "请安装 wget 或 curl"
        exit 1
    fi
}

download_gdrive_file() {
    local out="$1"
    local file_id="$2"
    if ! command -v gdown &> /dev/null; then
        echo "未检测到 gdown，尝试自动安装..."
        python -m pip install -U gdown || {
            echo "自动安装 gdown 失败，请手动执行: python -m pip install -U gdown"
            return 1
        }
    fi
    gdown "https://drive.google.com/uc?id=${file_id}" -O "$out"
}

echo "========== 下载到 $PRETRAINED =========="

# 1. yolo11l-seg.pt
if [ -f "yolo11l-seg.pt" ]; then
    echo "[删除] 旧的 yolo11l-seg.pt"
    rm -f "yolo11l-seg.pt"
fi
echo "[下载] yolo11l-seg.pt ..."
download yolo11l-seg.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt" || {
    echo "失败: yolo11l-seg.pt"; exit 1;
}

# 2. sam2.1_hiera_base_plus.pt
if [ -f "sam2.1_hiera_base_plus.pt" ]; then
    echo "[删除] 旧的 sam2.1_hiera_base_plus.pt"
    rm -f "sam2.1_hiera_base_plus.pt"
fi
echo "[下载] sam2.1_hiera_base_plus.pt ..."
download sam2.1_hiera_base_plus.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt" || {
    echo "失败: sam2.1_hiera_base_plus.pt"; exit 1;
}

# 3. depth_anything_v2_vitl.pth
if [ -f "depth_anything_v2_vitl.pth" ]; then
    echo "[删除] 旧的 depth_anything_v2_vitl.pth"
    rm -f "depth_anything_v2_vitl.pth"
fi
echo "[下载] depth_anything_v2_vitl.pth ..."
download depth_anything_v2_vitl.pth "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth" || {
    echo "失败: depth_anything_v2_vitl.pth"; exit 1;
}

# 4. droid.pth
if [ -f "droid.pth" ]; then
    echo "[删除] 旧的 droid.pth"
    rm -f "droid.pth"
fi
echo "[下载] droid.pth ..."
download droid.pth "https://huggingface.co/vslamlab/droidslam/resolve/main/droid.pth" || {
    echo "失败: droid.pth (可尝试从 Google Drive 下载: https://drive.google.com/file/d/1oZbVPrubtaIUjRRuT8F-YjjHBW-1spKT)"; exit 1;
}

# 5. raft-things.pth (光流 RAFT)
if [ -f "raft-things.pth" ]; then
    echo "[删除] 旧的 raft-things.pth"
    rm -f "raft-things.pth"
fi
echo "[下载] raft-things.pth (RAFT 光流) ..."
download raft-things.pth "https://huggingface.co/ddrfan/RAFT/resolve/main/raft-things.pth" || {
    echo "失败: raft-things.pth (可尝试: https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT)"; exit 1;
}

# 6. dsine.pt (法向 DSINE)
if [ -f "dsine.pt" ]; then
    echo "[删除] 旧的 dsine.pt"
    rm -f "dsine.pt"
fi
echo "[下载] dsine.pt (DSINE normal model from Google Drive) ..."
download_gdrive_file dsine.pt "1Wyiei4a-lVM6izjTNoBLIC5-Rcy4jnaC" || {
    echo "失败: dsine.pt"
    echo "可尝试手动下载目录: https://drive.google.com/drive/folders/1Wn83BXVXcErZZblUNgsT0k5IKR39dECe"
    echo "然后将 dsine.pt 放到 pretrained/"
    exit 1
}

echo "========== 全部下载完成 =========="
ls -la "$PRETRAINED"
