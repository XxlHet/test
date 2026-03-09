#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import datetime  # 新增：用于获取当前时间生成文件夹名

def find_latest_file(pattern):
    """全局搜索 result 文件夹，寻找最新生成的指定 CSV 文件"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    search_path = os.path.join(base_dir, 'result', '*', pattern)
    
    files = glob.glob(search_path)
    if not files:
        return None
    
    return max(files, key=os.path.getmtime)

def generate_comparison_plots(shape_name="sphere"):
    print(f"[*] 正在检索 '{shape_name}' 的最新实验数据...")
    
    # 分别独立查找最新的 Base 和 ATO 数据
    base_csv = find_latest_file(f"data_{shape_name}_Base.csv")
    ato_csv = find_latest_file(f"data_{shape_name}_ATO.csv")

    if not base_csv or not ato_csv:
        print("❌ 缺少对比文件！")
        if not base_csv:
            print(f"   -> 未能在任何结果目录中找到 'data_{shape_name}_Base.csv'")
        if not ato_csv:
            print(f"   -> 未能在任何结果目录中找到 'data_{shape_name}_ATO.csv'")
        return

    print(f"[*] 已加载 Baseline 数据: {base_csv}")
    print(f"[*] 已加载 ATO 数据: {ato_csv}")
    
    df_base = pd.read_csv(base_csv)
    df_ato = pd.read_csv(ato_csv)

    # 绘制的指标配置
    metrics = {
        'Target_Error(m)': ('Convergence Error Comparison', 'Mean Error (m)'),
        'Min_Distance(m)': ('Minimum Distance Comparison', 'Min Distance (m)'),
        'Avg_Velocity(m/s)': ('Average Velocity Comparison', 'Avg Velocity (m/s)')
    }

    # 学术论文配色规范
    color_base = '#E74C3C' # 红色系 (Baseline)
    color_ato = '#2ECC71'  # 绿色系 (ATO Ours)

    # ================= 新增：创建单独的时间戳文件夹 =================
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 获取当前时间并格式化为 YYYYMMDD_HHMMSS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 创建类似 result/20260309_163518_comparison 的文件夹
    new_target_dir = os.path.join(base_dir, 'result', f"{timestamp}_comparison")
    
    # exist_ok=True 确保即使文件夹存在也不会报错
    os.makedirs(new_target_dir, exist_ok=True)
    print(f"[*] 📁 创建图表输出独立目录: {new_target_dir}")
    # ==============================================================

    for col, (title, ylabel) in metrics.items():
        if col in df_base.columns and col in df_ato.columns:
            plt.figure(figsize=(9, 5.5))
            
            # Baseline: 较细的虚线
            plt.plot(df_base['Time(s)'], df_base[col], linewidth=1.5, color=color_base, linestyle='--', label='Baseline', alpha=0.8)
            
            # ATO (Ours): 较粗的实线
            plt.plot(df_ato['Time(s)'], df_ato[col], linewidth=2.5, color=color_ato, linestyle='-', label='ATO (Ours)')

            # 添加基准辅助线
            if col == 'Target_Error(m)':
                plt.axhline(y=0.0, color='black', linestyle=':', label='Ideal')
            elif col == 'Min_Distance(m)':
                plt.axhline(y=0.3, color='black', linestyle='-.', linewidth=1.5, label='Safety Limit (0.3m)')
                plt.axhspan(0, 0.3, color='gray', alpha=0.15)
                max_val = max(df_base[col].max() if not df_base[col].empty else 1.0, 
                              df_ato[col].max() if not df_ato[col].empty else 1.0)
                plt.ylim(bottom=0.25, top=max_val * 1.05)
            elif col == 'Avg_Velocity(m/s)':
                plt.axhline(y=1.0, color='blue', linestyle=':', alpha=0.5, label='Max Velocity')
                plt.ylim(bottom=-0.05, top=1.1)

            # 论文图表格式美化
            plt.title(title, fontweight='bold', fontsize=14)
            plt.xlabel('Time $t$ (s)', fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(loc='best', fontsize=11, frameon=True, shadow=True)
            
            # 保存高分辨率对比图到新文件夹
            save_name = f"Comparison_{shape_name}_{col.split('(')[0]}.png"
            plt.tight_layout()
            
            # 修改：将保存路径指向刚刚创建的时间戳文件夹
            save_path = os.path.join(new_target_dir, save_name)
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"✅ 成功生成图表: {save_path}")
        else:
            print(f"⚠️ 警告: 数据列 '{col}' 在 CSV 中缺失，已跳过该图表绘制。")

if __name__ == "__main__":
    import sys
    shape = "sphere"
    if len(sys.argv) > 1:
        shape = sys.argv[1]
    generate_comparison_plots(shape)