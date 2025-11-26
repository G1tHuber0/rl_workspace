#!/usr/bin/env python3
import torch
import argparse
import os

torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import gymnasium as gym
import time
import numpy as np
from stable_baselines3 import DQN, PPO, A2C,TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

# 尝试导入环境，兼容不同的目录结构
try:
    from turtlebot_env import TurtleBotEnv
except ImportError:
    # 如果找不到，尝试把当前目录加入路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from turtlebot_env import TurtleBotEnv

# ==========================================
# 1. 配置区域 (与您之前的配置保持一致)
# ==========================================

ALGOS = {
    "dqn": DQN,
    "ppo": PPO,
    "a2c": A2C,
}

HYPERPARAMS = {
    "dqn": {
        "policy": "MlpPolicy",
        "learning_rate": 7e-4, 
        "buffer_size": 100_000, # 并行后数据量大，加大经验池
        "learning_starts": 10_000, 
        "batch_size": 128,      
        "exploration_fraction": 0.4, 
        "exploration_final_eps": 0.05,
        "gamma": 0.99,            
        "target_update_interval": 1000,
        "gradient_steps" : 1,
    },
    "ppo": {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "ent_coef": 0.01,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
    },
    "a2c": {
        "policy": "MlpPolicy",
        "learning_rate": 7e-4,
        "n_steps": 20,
        "ent_coef": 0.01,
    }
}

def make_env(rank, seed=0):
    """
    用于创建环境的辅助函数，SubprocVecEnv 需要这个函数
    :param rank: 环境的索引 (0, 1, 2, ...) -> 对应 worker_id
    :param seed: 随机种子
    """
    def _init():
        # 传入 worker_id，触发 turtlebot_env.py 中的并行逻辑 (修改环境变量)
        env = TurtleBotEnv(worker_id=rank)
        # 设置不同的随机种子，保证每个环境的随机性不同
        env.reset(seed=seed + rank) 
        return env
    return _init

def get_args():
    parser = argparse.ArgumentParser(description="TurtleBot3 并行训练脚本")
    parser.add_argument("--algo", type=str, default="ppo", choices=ALGOS.keys())
    parser.add_argument("--steps", type=int, default=500000, help="训练总步数")
    parser.add_argument("--n_envs", type=int, default=6, help="并行环境数量 (需与 launch_parallel.py 一致)")
    parser.add_argument("--save_name", type=str, default="turtlebot_parallel")
    parser.add_argument("--load", type=str, default=None)
    return parser.parse_args()

def main():
    args = get_args()
    
    # 路径设置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "../logs")
    model_dir = os.path.join(script_dir, "../models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"🚀 [Train] 任务启动: 算法={args.algo.upper()}, 进程数={args.n_envs}, 总步数={args.steps}")

    # ==========================================
    # 2. 创建向量化环境 (核心修改)
    # ==========================================
    # SubprocVecEnv 会在后台创建 n_envs 个进程，每个进程运行一个 make_env 返回的环境
    # 这里的 i 对应 worker_id (0, 1, 2...)
    env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    
    # VecMonitor 用于记录每个 episode 的奖励和长度，方便 Tensorboard 显示
    # 它会自动统计所有并行环境的数据
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))

    # 3. 实例化/加载 模型
    if args.algo not in ALGOS: raise ValueError(f"Unknown algo: {args.algo}")
    AlgorithmClass = ALGOS[args.algo]
    
    if args.load:
        print(f"📥 正在加载模型: {args.load}")
        # 注意: load 时需要传入 env，模型会自动适应向量化环境
        model = AlgorithmClass.load(args.load, env=env, tensorboard_log=log_dir)
    else:
        print(f"✨ 创建新模型 ({args.algo})")
        algo_params = HYPERPARAMS.get(args.algo, {})
        
        final_kwargs = {
            "env": env,
            "verbose": 1,
            "tensorboard_log": log_dir,
            "device": "auto",
            **algo_params
        }
        model = AlgorithmClass(**final_kwargs)

    # 4. 开始训练
    start_time = time.time()
    try:
        print("⏳ 开始训练...")
        # total_timesteps 是所有环境加起来的总步数
        model.learn(total_timesteps=args.steps, progress_bar=True,log_interval=1)
        
        # 5. 保存模型
        save_path = os.path.join(model_dir, f"{args.algo}_{args.save_name}")
        model.save(save_path)
        print(f"✅ 训练完成! 耗时: {(time.time()-start_time)/60:.1f}分钟")
        print(f"💾 模型已保存: {save_path}.zip")
        
    except KeyboardInterrupt:
        print("\n⚠️ 手动中断，正在紧急保存...")
        save_path = os.path.join(model_dir, f"{args.algo}_{args.save_name}_interrupted")
        model.save(save_path)
        print(f"💾 紧急备份已保存: {save_path}.zip")
        
    finally:
        # 关闭所有并行环境进程
        env.close()
        print("[Train] 所有环境已关闭")

if __name__ == "__main__":
    main()