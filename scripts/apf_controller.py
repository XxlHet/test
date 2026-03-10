#!/usr/bin/env python3

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.neighbors import BallTree
from scipy.optimize import linear_sum_assignment  
import os
import csv
import time
import pandas as pd

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

class APFSwarmController():
    def __init__(self, p_cohesion=1.0, p_seperation=1.0, p_alignment=1.0, max_vel=0.5, min_dist=0.3) -> None:
        self.swarm = None
        self.goals = None
        
        self.min_dist = min_dist
        self.max_vel = max_vel
        
        self.velocities = None
        self.p_separation = p_seperation
        self.p_cohesion = p_cohesion

        self.enable_ato = False  
        
        self.log_dir = ""            
        self.current_log_name = ""   
        self.csv_initialized = False 
        self.start_time = 0.0        
        self.last_csv_path = ""      

        self.is_returning = False
        self.return_start_poses = None
        self.return_home_poses = None
        self.return_start_time = 0
        self.return_duration = 5.0
        
        # 🌟 动态记录当前活跃和构图的无人机数量，避免待命机群污染数据
        self.current_shape_num = 0
        self.current_active_num = 0

    def initiate_safe_return(self, start_poses, home_poses):
        """ 强制全员返航降落 (用于实验彻底结束) """
        self.is_returning = True
        n = min(len(start_poses), len(home_poses))
        self.return_start_poses = start_poses[:n].copy()
        self.return_home_poses = home_poses[:n].copy()
            
        max_dist = np.max(np.linalg.norm(self.return_home_poses - self.return_start_poses, axis=1))
        self.return_duration = max(max_dist / (self.max_vel * 0.45), 7.0) 
        
        self.return_start_time = time.time()
        self.goals = self.return_start_poses.copy() 
        
        print(f"\n[SRM] Full Fleet Safe Return Activated. Est. time: {self.return_duration:.1f}s")

    def distribute_goals(self, start, goals, shape_num=None, active_num=None):
        """
        🌟 终极 ATO 混合调度：支持半空中分离、补充与变阵
        """
        if shape_num is None: shape_num = len(goals)
        if active_num is None: active_num = len(goals)
        
        self.current_shape_num = shape_num
        self.current_active_num = active_num

        if self.enable_ato and active_num > 1:
            # 仅截取参与本次动态变化的无人机（包括待飞的、空中的、和要降落的）
            active_start = start[:active_num]
            active_goals = goals[:active_num]
            
            # 分离出构图目标与返航目标
            shape_goals = active_goals[:shape_num]
            rtb_goals = active_goals[shape_num:]

            # 仅对构图目标执行空间流形松弛，保护待命网格不被破坏
            if len(shape_goals) > 1:
                dist_matrix_llm = cdist(shape_goals, shape_goals)
                np.fill_diagonal(dist_matrix_llm, np.inf) 
                effective_min = np.percentile(np.min(dist_matrix_llm, axis=1), 10)
                if effective_min < 0.001: effective_min = 0.001 
                
                target_spacing = self.min_dist * 0.90 
                scale = min(max(target_spacing / effective_min, 1.0), 3.0) 
                
                centroid = np.mean(shape_goals, axis=0)
                scaled_shape_goals = centroid + (shape_goals - centroid) * scale

                for _ in range(50):
                    dists = cdist(scaled_shape_goals, scaled_shape_goals)
                    np.fill_diagonal(dists, np.inf)
                    min_dists = np.min(dists, axis=1)
                    if np.min(min_dists) >= target_spacing * 0.99: break
                        
                    displacement = np.zeros_like(scaled_shape_goals)
                    for i in range(len(scaled_shape_goals)):
                        mask = dists[i] < target_spacing
                        if np.any(mask):
                            vecs = scaled_shape_goals[i] - scaled_shape_goals[mask]
                            ds = dists[i][mask].reshape(-1, 1)
                            pushes = (vecs / ds) * (target_spacing - ds) * 0.5
                            displacement[i] = np.sum(pushes, axis=0)
                    scaled_shape_goals += displacement
            else:
                scaled_shape_goals = shape_goals

            # 将松弛后的构图目标与原地待命目标重新拼接
            if len(rtb_goals) > 0:
                scaled_active_goals = np.vstack((scaled_shape_goals, rtb_goals))
            else:
                scaled_active_goals = scaled_shape_goals

            # 全局拓扑解耦：匈牙利算法会自动判断谁飞往新图形最近，谁直接回家降落最近！
            dist_matrix = cdist(active_start, scaled_active_goals)
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            
            out_active_goals = np.zeros_like(scaled_active_goals)
            for r, c in zip(row_ind, col_ind):
                out_active_goals[r] = scaled_active_goals[c]
                
            self.goals = np.copy(goals)
            self.goals[:active_num] = out_active_goals
            print(f"\n[ATO] Optimal topology computed. Shape drones: {shape_num} | RTB/Standby: {active_num - shape_num}")
        else:
            dist_matrix = cdist(start, goals)
            out_goals = np.zeros_like(goals)
            for i in range(start.shape[0]):
                ind = np.argmin(dist_matrix[i])
                if i < len(out_goals):
                    out_goals[i] = goals[ind]
                dist_matrix[i, :] = np.inf
                dist_matrix[:, ind] = np.inf
            self.goals = out_goals

    def get_control(self, poses) -> None:
        n = min(self.goals.shape[0], poses.shape[0])
        poses = poses[:n]
        if self.velocities is None:
            self.velocities = np.zeros_like(poses)
            
        if self.is_returning:
            elapsed = time.time() - self.return_start_time
            progress = min(elapsed / self.return_duration, 1.0)
            smooth_p = progress * progress * (3 - 2 * progress) 
            bloom_scale = 1.0 + 1.2 * np.sin(np.pi * smooth_p) 
            centroid = np.mean(self.return_home_poses, axis=0)
            bloomed_home = centroid + (self.return_home_poses - centroid) * bloom_scale
            self.goals[:n] = self.return_start_poses + (bloomed_home - self.return_start_poses) * smooth_p

        ball_tree = BallTree(poses[:, :2], metric='euclidean')
        control_vels = np.zeros_like(poses)

        error_vec = self.goals[:n] - poses[:n]
        dist_to_goal = np.linalg.norm(error_vec, axis=1, keepdims=True)
        scaling = np.where(dist_to_goal < 0.05, dist_to_goal / 0.05, 1.0) 
        vel_cohesion = self.p_cohesion * error_vec * scaling

        for i, pose in enumerate(poses):
            query_pose = pose[:2]
            v_nom = vel_cohesion[i].copy()
            if np.linalg.norm(v_nom) > self.max_vel:
                v_nom = (v_nom / np.linalg.norm(v_nom)) * self.max_vel

            interaction_radius = self.min_dist * 2.0
            nearest_ind = ball_tree.query_radius(query_pose.reshape(1, -1), interaction_radius)[0][1:]
            
            v_rep = np.zeros(3)
            for ind in nearest_ind:
                p_rel = pose - poses[ind]
                dist = np.linalg.norm(p_rel)
                
                if dist < self.min_dist:
                    safe_dist = max(dist, 0.01)
                    repulsive_mag = self.p_separation * (1.0 / safe_dist - 1.0 / self.min_dist) / (safe_dist ** 2 + 0.01)
                    v_rep += repulsive_mag * (p_rel / safe_dist)
            
            v_rep[2] = 0
            control_vels[i] = v_nom + v_rep

        control_vels = 0.8 * control_vels + 0.2 * self.velocities[:n]
        for k in range(len(control_vels)):
            speed = np.linalg.norm(control_vels[k])
            if speed > self.max_vel:
                control_vels[k] = (control_vels[k] / speed) * self.max_vel
        self.velocities[:n] = control_vels.copy()

        # 🌟 仅对当前处于构图状态的无人机进行数据记录，屏蔽地面待机飞机的噪音
        if self.log_dir and self.current_log_name and self.current_shape_num > 0:
            full_path = os.path.join(self.log_dir, f"{self.current_log_name}.csv")
            
            if self.last_csv_path != full_path:
                self.csv_initialized = False
                self.last_csv_path = full_path

            if not self.csv_initialized:
                try:
                    with open(full_path, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(["Time(s)", "Min_Distance(m)", "Avg_Velocity(m/s)", "Target_Error(m)"])
                    self.start_time = time.time()
                    self.csv_initialized = True
                except Exception:
                    return control_vels

            curr_t = round(time.time() - self.start_time, 2)
            
            eval_poses = poses[:self.current_shape_num]
            eval_goals = self.goals[:self.current_shape_num]
            eval_vels = control_vels[:self.current_shape_num]
            
            if self.current_shape_num > 1:
                diffs = eval_poses[:, np.newaxis, :] - eval_poses[np.newaxis, :, :]
                dists = np.linalg.norm(diffs, axis=-1)
                np.fill_diagonal(dists, np.inf)
                min_d = round(np.min(dists), 4)
            else:
                min_d = 0.0
                
            avg_v = round(np.mean(np.linalg.norm(eval_vels, axis=1)), 4)
            err = round(np.mean(np.linalg.norm(eval_goals - eval_poses, axis=1)), 4)

            try:
                with open(full_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([curr_t, min_d, avg_v, err])
            except:
                pass 
        return control_vels

    def generate_plots(self):
        if not self.last_csv_path or not os.path.exists(self.last_csv_path): return
        
        mode_prefix = "ATO" if self.enable_ato else "Base"
        algo_label = "ATO (Ours)" if self.enable_ato else "Baseline"

        print(f"\n[*] Generating plots for [{algo_label}] mode...")
        try:
            df = pd.read_csv(self.last_csv_path)
            metrics = {
                'Target_Error(m)': ('Convergence Error Comparison', 'Mean Error (m)', '#2ECC71' if self.enable_ato else '#E74C3C'),
                'Min_Distance(m)': ('Minimum Distance Comparison', 'Min Distance (m)', '#2ECC71' if self.enable_ato else '#E74C3C'),
                'Avg_Velocity(m/s)': ('Average Velocity Comparison', 'Avg Velocity (m/s)', '#2ECC71' if self.enable_ato else '#E74C3C')
            }
            
            for col, (title, ylabel, color) in metrics.items():
                if col in df.columns:
                    plt.figure(figsize=(9, 5.5))
                    plt.plot(df['Time(s)'], df[col], linewidth=2.5 if self.enable_ato else 1.5, 
                             color=color, linestyle='-' if self.enable_ato else '--', 
                             label=algo_label, alpha=0.9)
                    
                    if col == 'Target_Error(m)':
                        plt.axhline(y=0.0, color='black', linestyle=':', label='Ideal')
                    elif col == 'Min_Distance(m)':
                        plt.axhline(y=self.min_dist, color='black', linestyle='-.', label=f'Safety Limit ({self.min_dist}m)')
                        plt.axhspan(0, self.min_dist, color='gray', alpha=0.15)
                        plt.ylim(bottom=max(0, self.min_dist - 0.05), top=df[col].max() * 1.05)
                    elif col == 'Avg_Velocity(m/s)':
                        plt.axhline(y=self.max_vel, color='blue', linestyle=':', alpha=0.5, label='Max Velocity')
                        plt.ylim(bottom=-0.05, top=self.max_vel + 0.1)

                    plt.title(title, fontweight='bold', fontsize=14)
                    plt.xlabel('Time $t$ (s)', fontsize=12)
                    plt.ylabel(ylabel, fontsize=12)
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.legend(loc='best', fontsize=11, frameon=True, shadow=True)
                    
                    img_name = f"{mode_prefix}_{self.current_log_name}_{col.split('(')[0]}.png"
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.log_dir, img_name), dpi=300)
                    plt.close()
            print(f"[*] Plots saved: {self.log_dir}")
        except Exception as e:
            print(f"⚠️ Plotting Error: {e}")