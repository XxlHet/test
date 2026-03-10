#!/usr/bin/env python3

from flock_gpt.msg import Vector3StampedArray
from geometry_msgs.msg import Vector3
import numpy as np
import rospy
from sdf import box, sphere, write_binary_stl, rounded_box, capsule
from sdf import generate
from apf_controller import APFSwarmController
from gpt_sdf import SDFDialog, SDFModel
import threading
import os
import time
from point_distributor import PointDistributer

class SwarmControllerNode():
    def __init__(self, goals=[]) -> None:
        print("\n" + "="*60)
        print("   🚀 FLOCK-GPT: Fleet Management System Initializing")
        print("="*60)
        
        # 🌟 核心理念：设定最大机队容量（无人机池）
        val = input(">>> Enter Maximum Fleet Capacity (e.g. 200, must be >= max shape size): ")
        try:
            self.fleet_capacity = int(val) if val.strip() else 200
        except:
            self.fleet_capacity = 200
            
        # 设置模拟器一次性生成，后续绝不修改此参数，彻底杜绝重启
        rospy.set_param('/swarm_num_drones', self.fleet_capacity)
        
        self.shape_drones = 0       # 当前形状所需无人机数
        self.prev_active_drones = 0 # 记录上一轮飞在空中的无人机数

        print("\n--- Algorithm Module Configuration ---")
        module_input = input(">>> Enable ATO (Adaptive Trajectory Optimization)? [y/N]: ")
        self.enable_ato = True if module_input.lower() == 'y' else False
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.save_dir = os.path.join(base_dir, 'result', timestamp)
        os.makedirs(self.save_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print(f"[*] Mode: ATO Enabled={self.enable_ato}")
        print(f"[*] Fleet Capacity: {self.fleet_capacity} drones (Simulating...)")
        print("="*60 + "\n")

        self.is_running = False 
        self.goals = []
        self.start_poses = None
        self.home_poses = None  
        self.trigger_return = False 
        
        self.controller = APFSwarmController(max_vel=1)
        self.controller.log_dir = self.save_dir 
        self.controller.enable_ato = self.enable_ato
        
        self.model = SDFModel()
        self.dialog = SDFDialog()

        rospy.Subscriber("/swarm/poses", Vector3StampedArray, self.callback_state, queue_size=1)
        self.cmd_vel_publisher = rospy.Publisher('/swarm/cmd_vel', Vector3StampedArray, queue_size=1)
        
        threading.Thread(target=self.continuous_input_prompt, daemon=True).start()

    def callback_state(self, msg:Vector3StampedArray):
        poses = np.array([[p.x, p.y, p.z] for p in msg.vector])
        
        # 🌟 获取模拟器中全部的机队网格坐标，作为永久的大本营
        if self.home_poses is None and len(poses) >= self.fleet_capacity:
            self.home_poses = poses[:self.fleet_capacity].copy()
            print(f"[*] Captured home grid for {self.fleet_capacity} drones. Fleet ready for dispatch.")
            
        if getattr(self, 'trigger_return', False):
            self.controller.initiate_safe_return(poses, self.home_poses)
            self.trigger_return = False
            self.start_poses = poses 
        
        if not self.is_running or self.goals is None or np.array(self.goals).size == 0:
            return
            
        if self.start_poses is None:
            if self.home_poses is None: return # 等待环境加载
            
            # 计算本次调度牵涉到的无人机总数 (当前需要的 vs 原本在天上需要返航的)
            active_drones = max(self.prev_active_drones, self.shape_drones)
            
            self.controller.distribute_goals(
                poses, 
                self.goals, 
                shape_num=self.shape_drones, 
                active_num=active_drones
            ) 
            self.start_poses = poses
            # 记录本次飞行的活跃状态，用于下一轮判断
            self.prev_active_drones = self.shape_drones

        vels = self.controller.get_control(poses)
        cmd_vel = Vector3StampedArray()
        
        for vel in vels:
            vect = Vector3()
            vect.x, vect.y, vect.z = vel[0], vel[1], vel[2]
            cmd_vel.vector.append(vect)
        
        if not rospy.is_shutdown():
            self.cmd_vel_publisher.publish(cmd_vel)

    def process_user_input(self, user_input):
        sdf_code = self.dialog.get_next_sdf_code(user_input)
        if sdf_code:
            local_vars = {"f": None}
            try: 
                f=10
                exec(sdf_code, globals(), local_vars)  
                f = local_vars.get("f")
                if f is not None:
                    pd_dist = PointDistributer(f)
                    points = pd_dist.generate_points(self.shape_drones)
                    
                    # 🌟 池化路由分配：构造全局目标阵列
                    full_goals = np.zeros((self.fleet_capacity, 3))
                    
                    # 1. 构图部队
                    if self.shape_drones > 0:
                        full_goals[:self.shape_drones] = points[:self.shape_drones]
                        
                    # 2. 待命/返航部队
                    if self.fleet_capacity > self.shape_drones:
                        full_goals[self.shape_drones:] = self.home_poses[self.shape_drones:]
                        
                    self.goals = full_goals
            except Exception as e:
                print("Error: ", e)

    def execute_return_sequence(self):
        if self.home_poses is None:
            print("🛑 Shutting down system...")
            rospy.signal_shutdown("User exit")
            return
            
        print("\n" + "="*60)
        print("[*] 🛬 Activating SRM (Safe Return Module)...")
        print("[*] Executing global return for all airborne drones.")
        print("="*60)
        
        self.trigger_return = True
        self.controller.current_log_name = "" 
        self.is_running = True
        
        input("\n>>> Press 'Enter' when all drones have landed safely to power off...")
        self.is_running = False
        print("🛑 System powered off successfully.")
        rospy.signal_shutdown("Experiment finished")

    def continuous_input_prompt(self):
        rospy.sleep(1.0)
        
        while not rospy.is_shutdown():
            # 动态请求无人机数量
            num_input = input(f"\n>>> Enter target shape size (Max {self.fleet_capacity}): ")
            try:
                self.shape_drones = min(int(num_input), self.fleet_capacity)
            except:
                print("[!] Invalid number. Setting to 10.")
                self.shape_drones = 10
                
            user_input = input(f">>> What to build with {self.shape_drones} drones? (e.g., sphere) [type 'exit' to quit]: \n")
            if user_input.lower() in ['exit', 'quit']:
                self.execute_return_sequence() 
                break
            
            shape_name = user_input.replace(" ", "_")
            mode_str = "ATO" if self.enable_ato else "Base"
            self.controller.current_log_name = f"data_{shape_name}_{self.shape_drones}drones_{mode_str}"
            
            self.controller.is_returning = False
            self.process_user_input(user_input)
            self.start_poses = None
            self.is_running = True 
            
            # 极其优雅的调度提示
            if self.shape_drones > self.prev_active_drones:
                print(f"[*] Scaling UP: {self.prev_active_drones} airborne + {self.shape_drones - self.prev_active_drones} launching from ground.")
            elif self.shape_drones < self.prev_active_drones:
                print(f"[*] Scaling DOWN: {self.shape_drones} morphing shape, {self.prev_active_drones - self.shape_drones} automatically returning to base.")
            else:
                print(f"[*] Seamless Morphing: All {self.shape_drones} airborne drones transitioning to new shape.")
                
            print(f"[*] Deploying '{shape_name}' (Log: {self.controller.current_log_name}.csv)...")

            input("\n>>> Press 'Enter' when formation is complete to generate individual plots...")
            self.is_running = False 
            
            self.controller.generate_plots()
            
            cont = input("\n>>> Do you want to try another shape? (y/n): ")
            if cont.lower() != 'y':
                self.execute_return_sequence() 
                break

            print("\n--- Algorithm Module Configuration (New Round) ---")
            module_input = input(f">>> Enable ATO (Adaptive Trajectory Optimization)? (Current: {self.enable_ato}) [y/N]: ")
            self.enable_ato = True if module_input.lower() == 'y' else False
            self.controller.enable_ato = self.enable_ato
            print(f"[*] Mode updated: ATO Enabled={self.enable_ato}")

            self.goals = []          
            self.start_poses = None  
            rospy.sleep(0.5) 

if __name__ == "__main__": 
    try:
        rospy.init_node('swarm_controller_node', anonymous=True)
        controller = SwarmControllerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass