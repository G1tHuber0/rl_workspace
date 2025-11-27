
import gymnasium as gym
import numpy as np
import rospy
import math
import random
import os  # 用于设置环境变量
from geometry_msgs.msg import Twist, Point, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker
from gymnasium import spaces
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion

import warnings
# 忽略特定的 Gym 警告
warnings.filterwarnings("ignore", category=UserWarning, module="gym")
warnings.filterwarnings("ignore", category=DeprecationWarning)

class TurtleBotEnv(gym.Env):
    def __init__(self, worker_id=None):
        super(TurtleBotEnv, self).__init__()

        # =================================================================
        # 【修改核心】兼容原来的方案 + 并行训练配置
        # =================================================================
        self.worker_id = worker_id
        
        if worker_id is not None:
            # === 模式 A: 并行训练模式 ===
            # 如果传入了 worker_id，说明是由 launch_parallel.py 启动的
            # 我们强制修改环境变量以连接到对应的隔离端口
            ros_port = 11311 + worker_id
            gazebo_port = 11345 + worker_id

            os.environ["ROS_MASTER_URI"] = f"http://localhost:{ros_port}"
            os.environ["GAZEBO_MASTER_URI"] = f"http://localhost:{gazebo_port}"
            
            # 使用带后缀的节点名，方便调试
            node_name = f'turtlebot_rl_env_{worker_id}'
            print(f"[Env Worker {worker_id}] 并行模式启动 -> 绑定端口 ROS: {ros_port}, Gazebo: {gazebo_port}")
            
        else:
            # === 模式 B: 原生兼容模式 (Original) ===
            # 如果没有传入 worker_id，说明是手动跑的单机训练
            # 不修改任何环境变量，完全信任当前终端的配置 (source devel/setup.bash)
            node_name = 'turtlebot_rl_env'
            print(f"[Env Single] 兼容模式启动 -> 使用当前环境变量 ROS_MASTER_URI: {os.environ.get('ROS_MASTER_URI', 'Default (11311)')}")

        # =================================================================

        # 1. 初始化 ROS 节点 (禁用信号处理以兼容 SB3)
        try:
            rospy.init_node(node_name, anonymous=True, disable_signals=True)
        except rospy.exceptions.ROSException:
            pass

        # 2. 定义通信接口
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.sub_odom = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # 可视化接口
        self.marker_pub = rospy.Publisher('/target_marker', Marker, queue_size=1)
        self.path_pub = rospy.Publisher('/train_path', Path, queue_size=5)
        
        # 服务接口 (只重置物理，不重置时间)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        # 3. 定义空间
        # 动作: 0=前进, 1=左转, 2=右转
        self.action_space = spaces.Discrete(3)
        
        # 观测: 24个雷达数据 + 1个目标距离 + 1个目标角度 = 26维
        # 所有数据均归一化到 [0, 1] 或 [-1, 1]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(26,), dtype=np.float32)

        # 4. 内部状态变量
        self.scan_data = np.ones(24) * 3.5
        self.position = Point()
        self.yaw = 0.0
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.current_dist = 0.0
        
        # 路径记录器
        self.path_record = Path()
        self.path_record.header.frame_id = "odom"

        # 5. 地图配置 (基于 SDF 文件解析的精确坐标)
        # 障碍物格式: (x, y, 禁区半径)
        # 实际半径0.15 + 机器人半径0.1 + 安全余量0.1 = 0.35
        self.obstacles = [
            # --- 左列 (x = -1.1) ---
            (-1.1, -1.1, 0.35),
            (-1.1,  0.0, 0.35),
            (-1.1,  1.1, 0.35),
            
            # --- 中列 (x = 0.0) ---
            (0.0, -1.1, 0.35),
            (0.0,  0.0, 0.35), # 地图中心
            (0.0,  1.1, 0.35),
            
            # --- 右列 (x = 1.1) ---
            (1.1, -1.1, 0.35),
            (1.1,  0.0, 0.35),
            (1.1,  1.1, 0.35),
        ]
        # 地图生成范围限制
        self.map_limit = 1.3 

    def scan_callback(self, msg):
        # 处理雷达数据：过滤、降采样、归一化
        raw = np.array(msg.ranges)
        # 将无穷大或NaN替换为最大距离
        raw = np.nan_to_num(raw, posinf=3.5, nan=3.5)
        # 过滤自身遮挡噪音 (小于0.12米视为误读)
        raw = np.where(raw < 0.12, 3.5, raw)
        
        # 降采样 360 -> 24
        # 注意: 确保这里不会越界，加个min保护
        step = max(1, len(raw)//24)
        indices = np.arange(0, len(raw), step)[:24]
        
        # 归一化到 [0, 1]
        self.scan_data = raw[indices] / 3.5

    def odom_callback(self, msg):
        self.position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, self.yaw) = euler_from_quaternion(orientation_list)

    def get_goal_info(self):
        # 计算目标距离
        goal_dist = math.sqrt((self.goal_x - self.position.x)**2 + (self.goal_y - self.position.y)**2)
        
        # 计算目标角度 (相对角度)
        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)
        heading = goal_angle - self.yaw
        
        # 角度归一化到 [-pi, pi]
        while heading > math.pi: heading -= 2 * math.pi
        while heading < -math.pi: heading += 2 * math.pi
        
        return goal_dist, heading

    def publish_marker(self):
        # 在 RViz 发布目标点 Marker
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0.1
        # 初始化四元数 (w=1 避免警告)
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2; marker.scale.y = 0.2; marker.scale.z = 0.2
        marker.color.a = 1.0; marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0
        self.marker_pub.publish(marker)

    def _check_valid_goal(self, x, y):
        # 检查1: 是否在圆柱体障碍物内
        for (ox, oy, radius) in self.obstacles:
            dist = math.sqrt((x - ox)**2 + (y - oy)**2)
            if dist < radius:
                return False
        
        # 检查2: 是否离机器人当前位置太近 (避免出生即胜利)
        # 假设机器人重置后在 (-2.0, -0.5) 附近 (根据launch文件)
        # 这里用更通用的逻辑：计算与当前位置的距离
        if math.sqrt((x - self.position.x)**2 + (y - self.position.y)**2) < 0.5:
            return False
            
        return True

    def step(self, action):
        # ------------------------------------------------------
        # 1. 执行动作 (Action Execution)
        # ------------------------------------------------------
        vel = Twist()
        
        # === 修改点 1：降低转弯速度，让动作更可控 ===
        # 原来是 2.0，太快了，容易这就好比让新手开法拉利
        angular_speed = 1.0 
        
        if action == 0:   # 前进
            vel.linear.x = 0.2
            vel.angular.z = 0.0
        elif action == 1: # 左转
            vel.linear.x = 0.05
            vel.angular.z = angular_speed
        elif action == 2: # 右转
            vel.linear.x = 0.05
            vel.angular.z = -angular_speed
        
        self.pub.publish(vel)
        # 这里的 sleep 在并行训练中可能不准确，主要靠 rospy 的频率控制
        # 如果是并行，建议注释掉或改极小，因为 SubprocVecEnv 会全速跑
        # rospy.sleep(0.05) 

        # ------------------------------------------------------
        # 2. 状态更新
       
        # 获取新的距离和角度
        dist, heading = self.get_goal_info()
        
        # 组装 Observation
        norm_dist = np.clip(dist / 3.5, 0, 1)
        norm_heading = heading / math.pi
        obs = np.concatenate((self.scan_data, [norm_dist, norm_heading])).astype(np.float32)

        # ------------------------------------------------------
        # 3. 奖励计算 (优化版)
        # ------------------------------------------------------
        reward = 0.0
        terminated = False
        min_laser = np.min(self.scan_data) 
        
        # A. 撞墙惩罚 (保持 -100)
        if min_laser < 0.05:
            reward = -200.0
            terminated = True
            # print("� 撞墙!") 
            self.pub.publish(Twist()) 
        
        # B. 抵达目标 (保持 100)
        elif dist < 0.15: # 稍微放宽一点点判定范围到 0.15
            reward = 200.0
            terminated = True
            # print("� 成功!")
            self.pub.publish(Twist()) 
            
        else:
            # === 修改点 2：进度奖励与朝向挂钩 ===
            # 只有当大致朝向目标时 (|heading| < 90度)，前进才给高分
            # 否则如果是背对着目标倒车(虽然我们没有倒车动作)或者乱跑，给分少
            progress = (self.current_dist - dist) * 10.0
            reward += progress
            
            self.current_dist = dist 

            # === 修改点 3：时间惩罚 ===
            reward -= 0.1
            
            # === 修改点 4：减轻避障势场 ===
            # 现在改成 * 3.0，稍微警告一下就行
            if min_laser < 0.2:
                reward -= (0.2 - min_laser) * 3.0
            
            # 4. 朝向奖励：删！或者给极小
            # 之前给 +0.05，它学会了原地转头骗分。
            # 现在改成：只有在真正前进的时候(action==0)，且方向对了，才给一点点
            if action == 0 and abs(heading) < 0.5:
                reward += 0.15
            
            # 额外惩罚：如果距离目标很远还转圈(不走)，重罚
            if action != 0:
                reward -= 0.05

        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. 物理重置
        rospy.wait_for_service('/gazebo/reset_world')
        try: self.reset_proxy()
        except: pass
        
        # 立即停车，防止带着旧速度冲出去
        self.pub.publish(Twist())

        # 2. 清空可视化路径
        self.path_record.poses.clear()
        self.path_pub.publish(self.path_record)

        # 3. 生成合法目标点 (拒绝采样)
        valid_goal = False
        while not valid_goal:
            tx = random.uniform(-self.map_limit, self.map_limit)
            ty = random.uniform(-self.map_limit, self.map_limit)
            if self._check_valid_goal(tx, ty):
                self.goal_x = tx
                self.goal_y = ty
                valid_goal = True
        
        self.publish_marker()
        
        # 4. 出生保护 (等待数据稳定)
        # 等待直到雷达数据不再显示“撞墙”
        # 有时候重置后，机器人可能会短暂地卡在之前的障碍物位置，需要等物理引擎弹开
        safe = False
        retry = 0
        while not safe and retry < 20:
            ##############################################################################rospy.sleep(0.1)
            # 检查是否有 > 0.1 (约35cm) 的空间
            if np.min(self.scan_data) > 0.1:
                safe = True
            retry += 1
            
        # 更新初始距离
        dist, heading = self.get_goal_info()
        self.current_dist = dist
        
        # 返回初始观测
        norm_dist = np.clip(dist / 3.5, 0, 1)
        norm_heading = heading / math.pi
        obs = np.concatenate((self.scan_data, [norm_dist, norm_heading])).astype(np.float32)
        
        return obs, {}