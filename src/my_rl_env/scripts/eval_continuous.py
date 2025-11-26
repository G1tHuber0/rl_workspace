#!/usr/bin/env python3
import argparse
import os
import time
import sys
from stable_baselines3 import DQN, PPO, A2C

try:
    from turtlebot_eval_env import TurtleBotEvalEnv # 导入新的评估环境
except ImportError:
    print("❌ 找不到 envs/turtlebot_eval_env.py")
    sys.exit(1)

IDE_CONFIG = {
    "algo": "ppo",
    "model_name": "ppo_turtlebot_model", # 换成你训练好的模型名
    "models_dir": "../models",
    "total_tasks": 50, # 也就是你要测试多少次“任务”
}

ALGOS = {"dqn": DQN, "ppo": PPO, "a2c": A2C}

def main():
    # ... (这里省略参数解析部分，和之前的一样，直接用 IDE_CONFIG) ...
    # 假设我们已经拿到了 model_path
    model_path = os.path.join(os.path.dirname(__file__), IDE_CONFIG["models_dir"], IDE_CONFIG["model_name"])
    if not os.path.exists(model_path + ".zip"): model_path += ".zip"

    print(f"🔄 加载连续评估环境...")
    env = TurtleBotEvalEnv() # 使用新的环境
    
    print(f"🔄 加载模型: {model_path}")
    model = ALGOS[IDE_CONFIG["algo"]].load(model_path, env=env)

    total_tasks = IDE_CONFIG["total_tasks"]
    print(f"🎬 开始连续导航测试 (共 {total_tasks} 个任务)")
    
    obs, _ = env.reset() # 第一次初始化
    
    finished_tasks = 0
    success_streak = 0 # 连胜纪录
    collision_count = 0
    
    while finished_tasks < total_tasks:
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                # 判断结果
                if reward > 100:
                    print(f"  ✅ 任务 {finished_tasks+1} 完成! 成功抵达。")
                    success_streak += 1
                    env.last_result = 'success' # 显式标记，虽然env内部也会判断
                else:
                    print(f"  💥 任务 {finished_tasks+1} 失败! 发生碰撞。")
                    success_streak = 0 # 连胜中断
                    collision_count += 1
                    env.last_result = 'crash'
                
                finished_tasks += 1
                
                # 只有当真正需要结束整个程序时才 break
                # 这里我们只是结束了一个“回合”，环境会在 reset 里处理连续逻辑
                done = True 
                
                # 稍微停顿观看效果
                time.sleep(1.0)
                
                # 关键：调用 reset 触发环境内部的 连续/重置 逻辑
                if finished_tasks < total_tasks:
                    obs, _ = env.reset()

    print("="*50)
    print(f"📊 测试结束")
    print(f"总任务: {total_tasks}")
    print(f"撞墙次数: {collision_count}")
    print(f"成功率: {(total_tasks - collision_count)/total_tasks*100:.1f}%")
    print("="*50)

if __name__ == "__main__":
    main()