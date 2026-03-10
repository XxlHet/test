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
        print("\n" + "="*50)
        val = input(">>> Please enter the initial number of drones (default 100): ")
        try:
            self.num_drones = int(val) if val.strip() else 100
        except:
            self.num_drones = 100
            
        rospy.set_param('/swarm_num_drones', self.num_drones)
        
        print("\n--- Algorithm Module Configuration ---")
        module_input = input(">>> Enable ATO (Adaptive Trajectory Optimization)? [y/N]: ")
        self.enable_ato = True if module_input.lower() == 'y' else False
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.save_dir = os.path.join(base_dir, 'result', timestamp)
        os.makedirs(self.save_dir, exist_ok=True)
        
        print("\n" + "="*50)
        print(f"[*] Mode: ATO Enabled={self.enable_ato}")
        print(f"[*] Drones: {self.num_drones} | Data Dir: {os.path.basename(self.save_dir)}")
        print("="*50 + "\n")

        self.is_running = False 
        self.goals = []
        self.start_poses = None
        self.home_poses = None  
        self.trigger_return = False # 🌟 SRM 触发器
        
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
        
        if self.home_poses is None or len(self.home_poses) != self.num_drones:
            if len(poses) == self.num_drones:
                self.home_poses = poses.copy()
        
        # 🌟 监听 SRM 触发信号
        if getattr(self, 'trigger_return', False):
            self.controller.initiate_safe_return(poses, self.home_poses)
            self.trigger_return = False
            self.start_poses = poses # 阻断下方的 distribute_goals
        
        if not self.is_running or self.goals is None or np.array(self.goals).size == 0:
            return
            
        if self.start_poses is None:
            self.controller.distribute_goals(poses, self.goals) 
            self.start_poses = poses

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
                    points = pd_dist.generate_points(self.num_drones)
                    self.goals = points
            except Exception as e:
                print("Error: ", e)

    def execute_return_sequence(self):
        if self.home_poses is None:
            print("🛑 Shutting down system...")
            rospy.signal_shutdown("User exit")
            return
            
        print("\n" + "="*50)
        print("[*] 🛬 Activating SRM (Safe Return Module)...")
        print("[*] Bypassing parking-lot paradox via cinematic morphing.")
        print("="*50)
        
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
            user_input = input("\n>>> What to build? (e.g., sphere, box) [type 'exit' to quit]: \n")
            if user_input.lower() in ['exit', 'quit']:
                self.execute_return_sequence() 
                break
            
            shape_name = user_input.replace(" ", "_")
            
            mode_str = "ATO" if self.enable_ato else "Base"
            self.controller.current_log_name = f"data_{shape_name}_{mode_str}"
            
            # 🌟 关闭返航状态，开始新的变形
            self.controller.is_returning = False
            self.process_user_input(user_input)
            self.start_poses = None
            self.is_running = True 
            print(f"[*] Command sent. Deploying '{shape_name}' (Log: {self.controller.current_log_name}.csv)...")

            input("\n>>> Press 'Enter' when formation is complete to generate individual plots...")
            self.is_running = False 
            
            self.controller.generate_plots()
            
            cont = input("\n>>> Do you want to try another experiment? (y/n): ")
            if cont.lower() != 'y':
                self.execute_return_sequence() 
                break
                
            new_num = input(f">>> Current drones: {self.num_drones}. Enter NEW number (or press Enter to keep): ")
            if new_num.strip():
                try:
                    self.num_drones = int(new_num)
                    rospy.set_param('/swarm_num_drones', self.num_drones)
                    self.home_poses = None 
                except:
                    pass

            print("\n--- Algorithm Module Configuration (New Round) ---")
            module_input = input(f">>> Enable ATO (Adaptive Trajectory Optimization)? (Current: {self.enable_ato}) [y/N]: ")
            self.enable_ato = True if module_input.lower() == 'y' else False
            self.controller.enable_ato = self.enable_ato
            print(f"[*] Mode updated: ATO Enabled={self.enable_ato}")

            rospy.set_param('/swarm_reset', True)
            self.goals = []          
            self.start_poses = None  
            print(f"[*] Resetting {self.num_drones} drones to initial ground positions...")
            rospy.sleep(1.0) 

if __name__ == "__main__": 
    try:
        rospy.init_node('swarm_controller_node', anonymous=True)
        controller = SwarmControllerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass