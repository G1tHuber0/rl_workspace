#!/usr/bin/env python3
import subprocess
import os
import time
import signal
import sys
import argparse


# ================= 配置 =================
DEFAULT_NUM_WORKERS = 1   # 默认并行环境数量
START_ROS_PORT = 11311    # 起始 ROS Master 端口
START_GAZEBO_PORT = 11345 # 起始 Gazebo 端口
LAUNCH_PKG = "my_rl_env"  # 你的包名
LAUNCH_FILE = "train_headless.launch" # 你的 launch 文件名
TRAIN_SCRIPT = "train.py" # 训练脚本的文件名
# =======================================

processes = []

def signal_handler(sig, frame):
    """清理函数：关闭所有子进程"""
    print("\n[Manager] 正在关闭所有环境...")
    for p in processes:
        try:
            # 发送 SIGTERM
            p.terminate()
            # 如果需要更强力的关闭，可以解开下面这行
            # p.kill() 
        except:
            pass
    
    # 确保也杀掉可能残留的 gzserver (可选，视情况而定)
    # subprocess.run(["killall", "-9", "gzserver", "gzclient"], stderr=subprocess.DEVNULL)
    
    print("[Manager] 所有环境已清理。")
    # 注意：这里不再直接 sys.exit(0)，而是让函数自然结束，
    # 这样可以在 try...finally 中被正确调用而不直接退解释器

def launch_environments():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="并行启动多个 ROS/Gazebo 环境并开始训练")
    parser.add_argument(
        "-n", "--num_workers", 
        type=int, 
        default=DEFAULT_NUM_WORKERS, 
        help=f"启动的环境数量 (默认: {DEFAULT_NUM_WORKERS})"
    )
    args = parser.parse_args()
    
    num_workers = args.num_workers

    # 注册 Ctrl+C 信号处理 (主要用于捕捉手动中断)
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0)) # 触发 finally 块

    print(f"[Manager] 准备启动 {num_workers} 个并行环境...")

    # --- 1. 启动 Gazebo/ROS 环境 ---
    for i in range(num_workers):
        ros_port = START_ROS_PORT + i
        gazebo_port = START_GAZEBO_PORT + i
        
        env = os.environ.copy()
        env["ROS_MASTER_URI"] = f"http://localhost:{ros_port}"
        env["GAZEBO_MASTER_URI"] = f"http://localhost:{gazebo_port}"
        
        cmd = [
            "roslaunch", 
            LAUNCH_PKG, 
            LAUNCH_FILE, 
            "-p", str(ros_port)
        ]
        
        print(f"[Manager] 启动 Worker {i} (ROS:{ros_port} | GZ:{gazebo_port})")
        
        # 启动 Gazebo 进程
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        processes.append(proc)
        
        # 错峰启动，防止 CPU 爆炸
        time.sleep(1) 

    print(f"[Manager] 所有环境启动完毕。等待 10 秒让 Gazebo 初始化...")
    time.sleep(10) # 关键：给 Gazebo 一点时间加载物理引擎

    # --- 2. 启动训练脚本 (train.py) ---
    # 获取 train.py 的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_script_path = os.path.join(current_dir, TRAIN_SCRIPT)

    # 构造训练命令，自动把环境数量传给 train.py
    train_cmd = [
        sys.executable, 
        "-u", 
        train_script_path,
        "--n_envs", str(num_workers) 
    ]

    print(f"🚀 [Manager] 启动主训练进程: {' '.join(train_cmd)}")

    try:
        # subprocess.run 会阻塞在这里，直到 train.py 运行结束
        subprocess.run(train_cmd, check=True)
        print("[Manager] 训练脚本执行完毕。")
    
    except subprocess.CalledProcessError as e:
        print(f"❌ [Manager] 训练脚本异常退出，错误码: {e.returncode}")
    except KeyboardInterrupt:
        print("\n🛑 [Manager] 用户手动中断。")
    finally:
        # --- 3. 无论训练成功还是失败，都执行清理 ---
        print("[Manager] 正在执行最终清理...")
        signal_handler(None, None)

if __name__ == "__main__":
    launch_environments()