import numpy as np
import rospy
import random
import math
import gymnasium as gym
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState # <--- 必须导入这个服务类型
from tf.transformations import quaternion_from_euler
# 继承你之前的训练环境
from turtlebot_env import TurtleBotEnv 

class TurtleBotEvalEnv(TurtleBotEnv):
    """
    评估专用环境：
    - 成功 -> 机器人原地继续，刷新目标
    - 失败 -> 机器人重置随机点，保留目标 (再试一次)
    """
    def __init__(self):
        super(TurtleBotEvalEnv, self).__init__()
        # 记录上个回合的结果: 'first', 'success', 'crash', 'timeout'
        self.last_result = 'first' 
        
        # === 修复 1: 显式定义服务接口 (防止父类没传过来) ===
        # 这是报错 'no attribute set_state' 的直接修复
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # === 修复 2: 防御性定义障碍物 (防止父类没定义) ===
        if not hasattr(self, 'obstacles'):
            self.obstacles = [
                (-1.1, -1.1, 0.35), (-1.1, 0.0, 0.35), (-1.1, 1.1, 0.35),
                (0.0, -1.1, 0.35), (0.0, 0.0, 0.35), (0.0, 1.1, 0.35),
                (1.1, -1.1, 0.35), (1.1, 0.0, 0.35), (1.1, 1.1, 0.35),
            ]
        if not hasattr(self, 'map_limit'):
            self.map_limit = 1.3

    # === 修复 3: 显式定义位置检查函数 ===
    def _is_valid_pos(self, x, y):
        """检查坐标是否在障碍物内"""
        for (ox, oy, radius) in self.obstacles:
            dist = math.sqrt((x - ox)**2 + (y - oy)**2)
            if dist < radius:
                return False
        return True

    def step(self, action):
        # 复用父类的 step 获取基本信息
        obs, reward, terminated, truncated, info = super().step(action)
        
        # === 关键：拦截 step 结果，记录死因 ===
        if terminated:
            if reward > 100:
                self.last_result = 'success'
            else:
                self.last_result = 'crash'
        elif truncated:
            self.last_result = 'timeout'
            
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = gym.utils.seeding.np_random(seed)

        print(f"\n[Eval Reset] 上局结果: {self.last_result} | ", end="")

        # 1. 清空路径和停止机器人
        self.path_record.poses.clear()
        self.path_pub.publish(self.path_record)
        self.pub.publish(Twist()) # 刹车

        # ==================================================
        # 分支 A: 第一次运行 (First)
        # ==================================================
        if self.last_result == 'first':
            print("初始化场景...")
            self._teleport_robot_random()
            self._spawn_goal_random_far()

        # ==================================================
        # 分支 B: 上局成功 (Success) -> 机器人不动，目标刷新
        # ==================================================
        elif self.last_result == 'success':
            print("机器人原地待命，生成新目标...")
            self._spawn_goal_random_far()

        # ==================================================
        # 分支 C: 上局撞墙/超时 (Crash/Timeout) -> 机器人重置，目标保留
        # ==================================================
        else:
            print("任务失败，机器人重置，目标保持不变...")
            self._teleport_robot_far_from_goal()

        # ------------------------------------------
        # 公共后续处理
        # ------------------------------------------
        self.publish_marker() # 更新 RViz 标记
        rospy.sleep(0.25)     # 等待数据刷新
        
        # 更新状态
        dist, heading = self.get_goal_info()
        self.current_dist = dist
        
        # 组装观测
        norm_dist = np.clip(dist / 3.5, 0, 1)
        norm_heading = heading / math.pi
        obs = np.concatenate((self.scan_data, [norm_dist, norm_heading])).astype(np.float32)
        
        return obs, {}

    # --- 辅助函数：瞬移机器人到随机位置 ---
    def _teleport_robot_random(self):
        valid = False
        rx, ry = 0, 0
        while not valid:
            rx = random.uniform(-self.map_limit, self.map_limit)
            ry = random.uniform(-self.map_limit, self.map_limit)
            if self._is_valid_pos(rx, ry):
                valid = True
        self._apply_teleport(rx, ry)

    # --- 辅助函数：瞬移机器人，但要离现有目标远一点 ---
    def _teleport_robot_far_from_goal(self):
        valid = False
        rx, ry = 0, 0
        while not valid:
            rx = random.uniform(-self.map_limit, self.map_limit)
            ry = random.uniform(-self.map_limit, self.map_limit)
            # 1. 位置本身合法
            if self._is_valid_pos(rx, ry):
                # 2. 离旧目标点距离 > 1.5m (防止重置在目标脸上)
                dist_to_goal = math.sqrt((rx - self.goal_x)**2 + (ry - self.goal_y)**2)
                if dist_to_goal > 1.5:
                    valid = True
        self._apply_teleport(rx, ry)

    # --- 辅助函数：生成随机目标，离机器人当前位置远 ---
    def _spawn_goal_random_far(self):
        valid = False
        gx, gy = 0, 0
        # 获取当前机器人位置
        rx, ry = self.position.x, self.position.y
        
        while not valid:
            gx = random.uniform(-self.map_limit, self.map_limit)
            gy = random.uniform(-self.map_limit, self.map_limit)
            # 1. 目标位置合法
            if self._is_valid_pos(gx, gy):
                # 2. 离机器人当前位置 > 1.5m
                dist = math.sqrt((gx - rx)**2 + (gy - ry)**2)
                if dist > 1.5:
                    self.goal_x = gx
                    self.goal_y = gy
                    valid = True

    # --- 底层：调用 Gazebo 服务 ---
    def _apply_teleport(self, x, y):
        state_msg = ModelState()
        state_msg.model_name = 'turtlebot3_burger'
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = 0.0
        
        yaw = random.uniform(-math.pi, math.pi)
        q = quaternion_from_euler(0, 0, yaw)
        state_msg.pose.orientation.x = q[0]
        state_msg.pose.orientation.y = q[1]
        state_msg.pose.orientation.z = q[2]
        state_msg.pose.orientation.w = q[3]
        
        try:
            rospy.wait_for_service('/gazebo/set_model_state')
            self.set_state(state_msg)
        except rospy.ServiceException as e:
            print(f"瞬移失败: {e}")