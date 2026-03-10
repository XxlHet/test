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

        # ===== 🌟 SRM (Safe Return Module) 核心参数 =====
        self.is_returning = False
        self.return_start_poses = None
        self.return_home_poses = None
        self.return_start_time = 0
        self.return_duration = 5.0

    def initiate_safe_return(self, start_poses, home_poses):
        """
        🌟 SRM: 绝对时间倒流 + 呼吸网格 (Breathing Grid)
        """
        self.is_returning = True
        n = min(len(start_poses), len(home_poses))
        
        # 强制第 i 架飞回第 i 个家，保留 1:1 绝对映射，绝不重新分配
        self.return_start_poses = start_poses[:n].copy()
        self.return_home_poses = home_poses[:n].copy()
            
        max_dist = np.max(np.linalg.norm(self.return_home_poses - self.return_start_poses, axis=1))
        # 预留充足的降落时间，让呼吸网格的展开与收缩动画极其优雅
        self.return_duration = max(max_dist / (self.max_vel * 0.45), 7.0) 
        
        self.return_start_time = time.time()
        self.goals = self.return_start_poses.copy() 
        
        print(f"\n[SRM] Safe Return Module Activated!")
        print(f"[*] Executing 'Breathing Grid' trajectory to eliminate APF walls. Est. time: {self.return_duration:.1f}s")

    def distribute_goals(self, start, goals):
        """ ATO 模块：空间流形松弛 + 拓扑解耦 """
        if self.enable_ato and len(goals) > 1:
            dist_matrix_llm = cdist(goals, goals)
            np.fill_diagonal(dist_matrix_llm, np.inf) 
            effective_min = np.percentile(np.min(dist_matrix_llm, axis=1), 10)
            if effective_min < 0.001: effective_min = 0.001 
            
            target_spacing = self.min_dist * 0.90 
            
            scale = target_spacing / effective_min
            scale = min(max(scale, 1.0), 3.0) 
            
            centroid = np.mean(goals, axis=0)
            scaled_goals = centroid + (goals - centroid) * scale

            # 空间流形松弛 (Spatial Manifold Relaxation)
            for _ in range(50):
                dists = cdist(scaled_goals, scaled_goals)
                np.fill_diagonal(dists, np.inf)
                min_dists = np.min(dists, axis=1)
                
                if np.min(min_dists) >= target_spacing * 0.99:
                    break
                    
                displacement = np.zeros_like(scaled_goals)
                for i in range(len(scaled_goals)):
                    mask = dists[i] < target_spacing
                    if np.any(mask):
                        vecs = scaled_goals[i] - scaled_goals[mask]
                        ds = dists[i][mask].reshape(-1, 1)
                        pushes = (vecs / ds) * (target_spacing - ds) * 0.5
                        displacement[i] = np.sum(pushes, axis=0)
                
                scaled_goals += displacement
            
            final_dists = cdist(scaled_goals, scaled_goals)
            np.fill_diagonal(final_dists, np.inf)
            final_min = np.min(final_dists)
            print(f"\n[ATO] Spatial Relaxation Complete. Safe min_dist: {final_min:.3f}m.")

            # 拓扑解耦
            dist_matrix = cdist(start, scaled_goals)
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            
            out_goals = np.zeros_like(scaled_goals)
            for r, c in zip(row_ind, col_ind):
                out_goals[r] = scaled_goals[c]
                
            self.goals = out_goals
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
        """ 底层 APF 引擎 """
        n = min(self.goals.shape[0], poses.shape[0])
        poses = poses[:n]
        if self.velocities is None:
            self.velocities = np.zeros_like(poses)
            
        # ======================================================================
        # 🌟 SRM 会呼吸的网格 (The Breathing Grid Algorithm)
        # ======================================================================
        if self.is_returning:
            elapsed = time.time() - self.return_start_time
            progress = min(elapsed / self.return_duration, 1.0)
            smooth_p = progress * progress * (3 - 2 * progress) # 0 到 1 的平滑进度
            
            # 核心魔法：使用正弦波让网格在飞行中途最高放大 2.2 倍，拉开巨大空隙
            bloom_scale = 1.0 + 1.2 * np.sin(np.pi * smooth_p) 
            
            # 计算动态呼吸目标点
            centroid = np.mean(self.return_home_poses, axis=0)
            bloomed_home = centroid + (self.return_home_poses - centroid) * bloom_scale
            
            self.goals[:n] = self.return_start_poses + (bloomed_home - self.return_start_poses) * smooth_p
        # ======================================================================

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

        # ======================================================================
        # 📊 数据记录逻辑：修复了 SyntaxError 语法截断的 bug
        # ======================================================================
        if self.log_dir and self.current_log_name:
            full_path = os.path.join(self.log_dir, f"{self.current_log_name}.csv")
            
            # 此处完整保留了判断条件
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
        """
        📈 恢复的图表生成逻辑：基于记录的阵型数据完美出图
        """
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