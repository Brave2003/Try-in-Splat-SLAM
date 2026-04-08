#!/bin/bash

# 定义源路径和目标路径的基础目录
SRC_BASE="/home/xuzhifeng/try/datasets/TUM_RGBD"
DST_BASE="/mnt/data/dataset_archive/zhaoyong/BONN"

# 遍历目标文件夹下所有以 rgbd_bonn_ 开头的目录（后面的 / 确保只匹配文件夹，不匹配 zip）
for dst_seq_dir in "$DST_BASE"/rgbd_bonn_*/; do
    
    # 提取 sequence 的名字，比如 rgbd_bonn_balloon
    seq_name=$(basename "$dst_seq_dir")
    
    # 定义具体的源路径和目标路径
    SRC_DIR="$SRC_BASE/$seq_name/normal"
    DST_DIR="$DST_BASE/$seq_name/normal/normal_npz"
    
    # 检查源文件夹是否存在
    if [ -d "$SRC_DIR" ]; then
        echo "正在处理: $seq_name"
        
        # 确保你这边的 normal 父级文件夹存在
        mkdir -p "$DST_BASE/$seq_name/normal"
        
        # 检查目标 normal_npz 是否已经存在，防止重复复制引发嵌套错误
        if [ ! -d "$DST_DIR" ]; then
            cp -r "$SRC_DIR" "$DST_DIR"
            echo "  [成功] 已复制到 -> $DST_DIR"
        else
            echo "  [跳过] $DST_DIR 已经存在"
        fi
    else
        echo "[警告] 找不到源文件夹: $SRC_DIR"
    fi
done

echo "全部处理完成！"