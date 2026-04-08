import subprocess
import time
import os
from queue import Queue

# --- 配置区域 ---
# 将你的配置文件路径填入此列表
CONFIG_FILES = [
    # "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/bonn_balloon2.yaml",
    # "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/bonn_person_track.yaml",
    # "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/bonn_person_track2.yaml",
    # "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/bonn_synchronous2.yaml",
    # "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/placing_box.yaml",
    # "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/placing_box2.yaml",
    # "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/placing_box3.yaml",
    "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/rgbd_dataset_freiburg3_sitting_rpy.yaml",
    "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/rgbd_dataset_freiburg3_sitting_static.yaml",
    "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/rgbd_dataset_freiburg3_sitting_xyz.yaml",
    "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/rgbd_dataset_freiburg3_walking_rpy.yaml",
    "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/rgbd_dataset_freiburg3_walking_static.yaml",
    "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/rgbd_dataset_freiburg3_walking_xyz.yaml"
    # "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/bonn_balloon.yaml",
]

CONFIG_FILES1 = [
    "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/bonn_balloon.yaml",
    "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/bonn_balloon2.yaml",
    "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/bonn_person_track.yaml",
    "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/bonn_person_track2.yaml",
    "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/bonn_synchronous2.yaml",
    "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/placing_box.yaml",
    "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/placing_box2.yaml",
    "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/placing_box3.yaml",
    "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/rgbd_dataset_freiburg3_sitting_rpy.yaml",
    "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/rgbd_dataset_freiburg3_sitting_static.yaml",
    "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/rgbd_dataset_freiburg3_sitting_xyz.yaml",
    "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/rgbd_dataset_freiburg3_walking_rpy.yaml",
    "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/rgbd_dataset_freiburg3_walking_static.yaml",
    "/home/zhaoyong/code/Try-in-Splat-SLAM/configs/TUM_RGBD/rgbd_dataset_freiburg3_walking_xyz.yaml"
]

LOG_DIR = "log/experiments"
CHECK_INTERVAL = 60  # 每 10 秒检测一次显卡状态
GPU_IDS = [0, 1, 3]     # 你想要使用的显卡索引（例如有两张卡就填 [0, 1]）

# ----------------

def get_gpu_processes():
    """检测所有显卡的进程数量"""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
            encoding='utf-8'
        )
        # 获取正在运行的 GPU UUIDs
        active_gpus = output.strip().split('\n')
        # 获取所有 GPU 的 UUID 映射
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            encoding='utf-8'
        ).strip().split('\n')
        
        gpu_map = {}
        for line in gpu_info:
            idx, uuid = line.split(', ')
            gpu_map[int(idx)] = uuid
            
        # 统计每个索引对应的进程数
        status = {idx: 0 for idx in GPU_IDS}
        for uuid in active_gpus:
            for idx, g_uuid in gpu_map.items():
                if uuid == g_uuid and idx in GPU_IDS:
                    status[idx] += 1
        return status
    except subprocess.CalledProcessError:
        # 如果 nvidia-smi 报错（可能暂无进程），返回全 0
        return {idx: 0 for idx in GPU_IDS}

def run_task_tracking():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    task_queue = Queue()
    for f in CONFIG_FILES:
        task_queue.put(f)

    active_jobs = {idx: None for idx in GPU_IDS} # 记录每个 GPU 上的进程对象

    print(f"🚀 开始调度任务，共 {task_queue.qsize()} 个任务...")

    while not task_queue.empty() or any(p is not None for p in active_jobs.values()):
        gpu_status = get_gpu_processes()

        for idx in GPU_IDS:
            # 检查当前 GPU 是否真的空闲（系统层面 + 我们自己记录的进程状态）
            if gpu_status[idx] == 0 and active_jobs[idx] is None:
                if not task_queue.empty():
                    config_path = task_queue.get()
                    scene_name = os.path.basename(config_path).replace(".yaml", "")
                    log_path = os.path.join(LOG_DIR, f"{scene_name}.log")

                    cmd = f"python run.py {config_path} --only_tracking"
                    
                    print(f"🔥 [GPU {idx}] 启动任务: {scene_name}")
                    
                    # 使用 Popen 异步启动，模拟 nohup 行为
                    with open(log_path, "w") as log_file:
                        proc = subprocess.Popen(
                            cmd,
                            shell=True,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(idx)},
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            preexec_fn=os.setpgrp # 保证脱离父进程组
                        )
                    active_jobs[idx] = proc

        # 检查已启动的任务是否完成，释放占位
        for idx in GPU_IDS:
            if active_jobs[idx] is not None:
                if active_jobs[idx].poll() is not None: # 进程已结束
                    print(f"✅ [GPU {idx}] 任务完成。")
                    active_jobs[idx] = None

        time.sleep(CHECK_INTERVAL)

    print("🏁 所有任务已处理完毕。")
    
def run_task():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    task_queue = Queue()
    for f in CONFIG_FILES1:
        task_queue.put(f)

    active_jobs = {idx: None for idx in GPU_IDS} # 记录每个 GPU 上的进程对象

    print(f"🚀 开始调度任务，共 {task_queue.qsize()} 个任务...")

    while not task_queue.empty() or any(p is not None for p in active_jobs.values()):
        gpu_status = get_gpu_processes()

        for idx in GPU_IDS:
            # 检查当前 GPU 是否真的空闲（系统层面 + 我们自己记录的进程状态）
            if gpu_status[idx] == 0 and active_jobs[idx] is None:
                if not task_queue.empty():
                    config_path = task_queue.get()
                    scene_name = os.path.basename(config_path).replace(".yaml", "")
                    log_path = os.path.join(LOG_DIR, f"{scene_name}.log")

                    cmd = f"python run.py {config_path}"
                    
                    print(f"🔥 [GPU {idx}] 启动任务: {scene_name}")
                    
                    # 使用 Popen 异步启动，模拟 nohup 行为
                    with open(log_path, "w") as log_file:
                        proc = subprocess.Popen(
                            cmd,
                            shell=True,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(idx)},
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            preexec_fn=os.setpgrp # 保证脱离父进程组
                        )
                    active_jobs[idx] = proc

        # 检查已启动的任务是否完成，释放占位
        for idx in GPU_IDS:
            if active_jobs[idx] is not None:
                if active_jobs[idx].poll() is not None: # 进程已结束
                    print(f"✅ [GPU {idx}] 任务完成。")
                    active_jobs[idx] = None

        time.sleep(CHECK_INTERVAL)

    print("🏁 所有任务已处理完毕。")
    

if __name__ == "__main__":
    # run_task_tracking()
    run_task()
