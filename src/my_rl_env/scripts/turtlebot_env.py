
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
        # 稍微降低线速度，提高角速度，增加灵活性
        if action == 0:   
            vel.linear.x = 0.2
            vel.angular.z = 0.0
        elif action == 1: # 左转
            vel.linear.x = 0.05
            vel.angular.z = 2.0 
        elif action == 2: # 右转
            vel.linear.x = 0.05
            vel.angular.z = -2.0
        
        self.pub.publish(vel)
        
        # 注意: rospy.sleep 会依赖于仿真时间。
        # 在并行训练中，只要每个 Gazebo 实例都在发布 /clock，这个 sleep 就会正常工作。
        ##########################################################################rospy.sleep(0.05) 

        # ------------------------------------------------------
        # 2. 状态更新 (State Update)
        # ------------------------------------------------------
        # 记录路径用于可视化
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "odom"
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose.position = self.position
        pose_stamped.pose.orientation.w = 1.0
        self.path_record.poses.append(pose_stamped)
        if len(self.path_record.poses) > 500: self.path_record.poses.pop(0)
        self.path_pub.publish(self.path_record)

        # 获取新的距离和角度
        dist, heading = self.get_goal_info()
        
        # 组装 Observation
        norm_dist = np.clip(dist / 3.5, 0, 1)
        norm_heading = heading / math.pi
        obs = np.concatenate((self.scan_data, [norm_dist, norm_heading])).astype(np.float32)

        # ------------------------------------------------------
        # 3. 奖励计算 (Reward Engineering) - 核心部分！
        # ------------------------------------------------------
        reward = 0.0
        terminated = False
        min_laser = np.min(self.scan_data) # 归一化后的雷达最小值 (0~1)
        
        # === A. 关键：进度奖励 (Progress Reward) ===
        # 这一步是让它学会“趋利”的关键。
        # 靠近了给正分，远离了给负分。放大系数 30 倍，让它比生存奖励更有吸引力。
        # 必须在判断撞墙/到达之前计算
        reward += (self.current_dist - dist) * 40.0
        self.current_dist = dist # 更新上一次的距离

        # === B. 撞墙惩罚 (Collision) ===
        # 0.06 * 3.5 ≈ 0.21m (机器人半径约0.1m，预留0.1m缓冲)
        if min_laser < 0.05:
            reward = -50.0
            terminated = True
            # 如果是并行训练，可以注释掉 print 以免刷屏
            # debug_id = self.worker_id if self.worker_id is not None else "Single"
            # print(f"[{debug_id}] 💥 撞墙! 距离目标: {dist:.2f}m")
            self.pub.publish(Twist()) # 停车
        
        # === C. 抵达目标 (Success) ===
        elif dist < 0.1:
            reward = 100.0
            terminated = True
            # debug_id = self.worker_id if self.worker_id is not None else "Single"
            # print(f"[{debug_id}] 🎉 成功! 目标:({self.goal_x:.2f}, {self.goal_y:.2f})")
            self.pub.publish(Twist()) # 停车
            
        else:
            # === D. 时间惩罚 (Time Penalty) ===
            # 强迫它走直线，不要磨蹭，不要原地转圈
            reward -= 0.05
            
            # === E. 避障势场 (Danger Penalty) ===
            # 当距离障碍物 < 0.5米 (0.15 * 3.5) 时
            # 距离越近，扣分越狠。这能教会它“贴墙走可以，但别太近”
            if min_laser < 0.15:
                # 扣分范围: 0 ~ -0.75
                reward -= (0.15 - min_laser) * 15.0
            
            # === F. 朝向奖励 (Heading Reward) - 可选 ===
            # 鼓励它把头对准目标，减小搜索空间
            # 如果朝向偏差 < 45度 (0.25 * pi)
            if abs(heading) < 0.2: # 对得很准
                reward += 0.1
            elif abs(heading) > 1.5: # 背对目标
                reward -= 0.1

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