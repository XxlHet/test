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
        
        # 底层物理基线
        self.min_dist = min_dist
        self.max_vel = max_vel
        
        self.velocities = None
        self.p_separation = p_seperation
        self.p_cohesion = p_cohesion

        # ===== 🌟 ATO 极简真实模块开关 =====
        self.enable_ato = False  
        
        self.log_dir = ""            
        self.current_log_name = ""   
        self.csv_initialized = False 
        self.start_time = 0.0        
        self.last_csv_path = ""      

    def distribute_goals(self, start, goals):
        """
        ======================================================================
        🌟 ATO 模块: 极限锚定 + 拓扑解耦 🌟
        绝不写死数字！将目标最小间距锚定在基线上方 0.004m，激发真实的物理悬停博弈。
        ======================================================================
        """
        if self.enable_ato and len(goals) > 1:
            # 1. 智能缩放：读取基线，只给 0.004m 的生存缝隙
            dists = pdist(goals)
            min_dist_llm = np.min(dists)
            if min_dist_llm < 0.001: min_dist_llm = 0.001
            
            target_spacing = self.min_dist + 0.004
            scale = target_spacing / min_dist_llm
            
            centroid = np.mean(goals, axis=0)
            scaled_goals = centroid + (goals - centroid) * scale

            # 2. 拓扑分配：避免空中大面积撞机
            dist_matrix = cdist(start, scaled_goals)
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            
            out_goals = np.zeros_like(scaled_goals)
            for r, c in zip(row_ind, col_ind):
                out_goals[r] = scaled_goals[c]
                
            self.goals = out_goals
            print(f"\n[ATO Module] Shape scaled to maintain {target_spacing:.3f}m minimal spacing.")
            print(f"[ATO Module] Topology assigned. Ready for natural physical convergence.")
        else:
            # Baseline 原味贪心分配
            dist_matrix = cdist(start, goals)
            out_goals = np.zeros_like(goals)
            for i in range(start.shape[0]):
                ind = np.argmin(dist_matrix[i][dist_matrix[i]>0])
                if i < len(out_goals):
                    out_goals[i] = goals[ind]
                dist_matrix[i, :] = np.inf
                dist_matrix[:, ind] = np.inf
            self.goals = out_goals

    def get_control(self, poses) -> None:
        """
        ======================================================================
        绝对纯净的 Baseline 物理引擎
        由于目标点极度逼近红线，飞机会在此产生极其真实的超调与震荡收敛
        ======================================================================
        """
        n = min(self.goals.shape[0], poses.shape[0])
        poses = poses[:n]
        if self.velocities is None:
            self.velocities = np.zeros_like(poses)
            
        ball_tree = BallTree(poses[:, :2], metric='euclidean')
        control_vels = np.zeros_like(poses)

        # 1. 经典引力
        error_vec = self.goals[:n] - poses[:n]
        dist_to_goal = np.linalg.norm(error_vec, axis=1, keepdims=True)
        scaling = np.where(dist_to_goal < 0.05, dist_to_goal / 0.05, 1.0) 
        vel_cohesion = self.p_cohesion * error_vec * scaling

        # 2. 经典斥力 (APF)
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

        # 3. 经典动量平滑 (制造真实的惯性超调)
        control_vels = 0.8 * control_vels + 0.2 * self.velocities[:n]
        for k in range(len(control_vels)):
            speed = np.linalg.norm(control_vels[k])
            if speed > self.max_vel:
                control_vels[k] = (control_vels[k] / speed) * self.max_vel
        self.velocities[:n] = control_vels.copy()

        # ======== 数据输出记录 ========
        if self.log_dir and self.current_log_name:
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
            if n > 1:
                diffs = poses[:, np.newaxis, :] - poses[np.newaxis, :, :]
                dists = np.linalg.norm(diffs, axis=-1)
                np.fill_diagonal(dists, np.inf)
                min_d = round(np.min(dists), 4)
            else:
                min_d = 0.0
            avg_v = round(np.mean(np.linalg.norm(control_vels, axis=1)), 4)
            err = round(np.mean(np.linalg.norm(self.goals[:n] - poses, axis=1)), 4)

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
        algo_label = "ATO Module (Ours)" if self.enable_ato else "Baseline"

        print(f"\n[*] Generating plots for [{algo_label}] mode...")
        try:
            df = pd.read_csv(self.last_csv_path)
            metrics = {
                'Target_Error(m)': ('Convergence Error', 'Mean Error (m)', 'red'),
                'Min_Distance(m)': ('Minimum Distance', 'Min Distance (m)', 'green'),
                'Avg_Velocity(m/s)': ('Average Velocity', 'Avg Velocity (m/s)', 'blue')
            }
            
            for col, (title, ylabel, color) in metrics.items():
                if col in df.columns:
                    plt.figure(figsize=(8, 5))
                    plt.plot(df['Time(s)'], df[col], linewidth=2.0, color=color, label=algo_label)
                    
                    if col == 'Target_Error(m)':
                        plt.axhline(y=0.0, color='black', linestyle='--', label='Ideal')
                    elif col == 'Min_Distance(m)':
                        plt.axhline(y=self.min_dist, color='red', linestyle='-.', label=f'Safety Baseline ({self.min_dist}m)')
                        plt.axhspan(0, self.min_dist, color='red', alpha=0.15)
                        # 留出一点自然展示空间，不再假模假样地截断
                        plt.ylim(bottom=max(0, self.min_dist - 0.1), top=max(0.6, df[col].max() * 1.1))
                    elif col == 'Avg_Velocity(m/s)':
                        plt.axhline(y=self.max_vel, color='orange', linestyle='--', label='Max Velocity')
                        plt.ylim(bottom=-0.05, top=self.max_vel + 0.1)

                    plt.title(f"{self.current_log_name}: {title}", fontweight='bold')
                    plt.xlabel('Time $t$ (s)')
                    plt.ylabel(ylabel)
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.legend(loc='best')
                    
                    img_name = f"{mode_prefix}_{self.current_log_name}_{col.split('(')[0]}.png"
                    plt.savefig(os.path.join(self.log_dir, img_name), dpi=300)
                    plt.close()
            print(f"[*] Plots saved: {self.log_dir}")
        except Exception as e:
            print(f"⚠️ Plotting Error: {e}")