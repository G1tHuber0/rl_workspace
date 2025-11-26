#!/usr/bin/env python3
import argparse
import os
import time
import sys
import numpy as np
from stable_baselines3 import DQN, PPO, A2C

# ==============================================================================
# 🔧 [IDE 配置区] - 在编辑器里直接修改这里即可一键运行
# ==============================================================================
IDE_CONFIG = {
    "algo": "ppo",               # 算法类型: "dqn", "ppo", "a2c"
    "model_name": "ppo_turtlebot_model",     # 模型文件名 (不需要带 .zip，会自动补全)
    "models_dir": "../models",   # 模型所在的文件夹 (相对路径或绝对路径)
    "episodes": 50,              # 测试的回合数
    "render": True,              # 是否打印详细日志
}
# ==============================================================================

# 尝试导入环境
try:
    from turtlebot_env import TurtleBotEnv
except ImportError:
    try:
        from turtlebot_env import TurtleBotEnv
    except ImportError:
        print("❌ 错误: 找不到 turtlebot_env.py，请确认路径正确")
        sys.exit(1)

ALGOS = {
    "dqn": DQN,
    "ppo": PPO,
    "a2c": A2C,
}

def get_args():
    """
    解析命令行参数。
    注意：所有参数默认值设为 None，以便我们判断用户是否在命令行输入了参数。
    """
    parser = argparse.ArgumentParser(description="TurtleBot3 模型评估脚本")
    
    parser.add_argument("--algo", type=str, default=None, choices=ALGOS.keys(), 
                        help="算法类型 (覆盖 IDE 配置)")
    
    parser.add_argument("--name", type=str, default=None, 
                        help="模型文件名 (覆盖 IDE 配置)")
    
    parser.add_argument("--dir", type=str, default=None, 
                        help="模型所在目录 (覆盖 IDE 配置)")
    
    parser.add_argument("--path", type=str, default=None, 
                        help="[高级] 直接指定模型的完整绝对路径 (优先级最高)")
    
    parser.add_argument("--episodes", type=int, default=None, 
                        help="测试回合数")

    return parser.parse_args()

def get_model_path(args):
    """
    智能路径解析逻辑：
    优先级: CLI --path > CLI --name > IDE_CONFIG
    """
    # 1. 如果命令行直接指定了完整路径 (--path)
    if args.path:
        target = args.path
    
    # 2. 否则，根据 --name/--dir 或者 IDE_CONFIG 拼接路径
    else:
        # 优先用命令行的，没有则用 IDE_CONFIG 的
        name = args.name if args.name else IDE_CONFIG["model_name"]
        directory = args.dir if args.dir else IDE_CONFIG["models_dir"]
        
        # 处理相对路径
        if not os.path.isabs(directory):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            directory = os.path.join(current_dir, directory)
            
        target = os.path.join(directory, name)

    # 3. 自动补全 .zip 后缀
    if not os.path.exists(target):
        if os.path.exists(target + ".zip"):
            target += ".zip"
    
    return target

def main():
    args = get_args()
    
    # === 参数合并逻辑 ===
    # 如果命令行没输，就用 IDE_CONFIG 里的值
    algo_name = args.algo if args.algo else IDE_CONFIG["algo"]
    n_episodes = args.episodes if args.episodes else IDE_CONFIG["episodes"]
    
    # 获取最终模型路径
    model_path = get_model_path(args)

    print("=" * 60)
    print(f"🤖 任务配置:")
    print(f"   - 算法: {algo_name.upper()}")
    print(f"   - 回合: {n_episodes}")
    print(f"   - 路径: {model_path}")
    print("=" * 60)

    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件!")
        print(f"   请检查路径是否正确: {model_path}")
        return

    # 1. 实例化环境
    print("🔄 初始化环境...", end="", flush=True)
    env = TurtleBotEnv()
    print(" [完成]")

    # 2. 加载模型
    print("🔄 加载模型权重...", end="", flush=True)
    AlgorithmClass = ALGOS[algo_name]
    try:
        model = AlgorithmClass.load(model_path, env=env)
        print(" [完成]")
    except ValueError:
        print("\n❌ 加载失败! 算法类型不匹配。")
        print(f"   你选择的是 {algo_name}，但模型文件可能不是用这个算法训练的。")
        return
    except Exception as e:
        print(f"\n❌ 加载出错: {e}")
        return

    # 3. 开始测试
    print(f"\n🎬 开始测试 (共 {n_episodes} 回合)")
    
    success_count = 0
    crash_count = 0
    timeout_count = 0
    
    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        
        print(f"\n🔹 Episode {ep}/{n_episodes} | 目标: ({env.goal_x:.2f}, {env.goal_y:.2f})")
        
        while not done:
            # 确定性预测 (关闭随机探索)
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            done = terminated or truncated
            
            if terminated:
                if reward > 100:
                    print(f"   ✅ 成功到达! (步数: {step_count} | Reward: {total_reward:.2f})")
                    success_count += 1
                elif reward < -100:
                    print(f"   💥 发生碰撞! (步数: {step_count} | Reward: {total_reward:.2f})")
                    crash_count += 1
                else:
                    print(f"   ⏹️ 其他结束. (Reward: {total_reward:.2f})")
            
            if step_count > 1000:
                print("   ⏳ 超时强制结束")
                timeout_count += 1
                done = True

        # 回合间停顿
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("📊 测试总结报告")
    print(f"   - 总回合数: {n_episodes}")
    print(f"   - 成功次数: {success_count} ({success_count/n_episodes*100:.1f}%)")
    print(f"   - 碰撞次数: {crash_count} ({crash_count/n_episodes*100:.1f}%)")
    print(f"   - 超时次数: {timeout_count}")
    print("=" * 60)

if __name__ == "__main__":
    main()