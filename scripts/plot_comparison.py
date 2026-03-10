#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import datetime

def get_all_csv_files(limit=30):
    """全局搜索 result 文件夹，按生成时间倒序列出最近的 CSV 文件"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    search_path = os.path.join(base_dir, 'result', '**', '*.csv')
    
    # 使用 recursive=True 查找所有子目录下的 csv
    files = glob.glob(search_path, recursive=True)
    if not files:
        return []
    
    # 按修改时间倒序排列（最新的在最前）
    files.sort(key=os.path.getmtime, reverse=True)
    return files[:limit] # 默认只显示最近的 30 个避免刷屏

def generate_multi_comparison_plots():
    print("\n" + "="*60)
    print("   📊 Multi-Agent Swarm Data Comparison Tool")
    print("="*60)

    # 1. 检索并展示文件
    csv_files = get_all_csv_files()
    if not csv_files:
        print("❌ 未在 result 目录中找到任何 CSV 数据文件！")
        return

    print("\n[*] 发现以下最近的实验数据：")
    for i, f in enumerate(csv_files):
        # 提取相对路径，方便查看
        rel_path = os.path.relpath(f, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        print(f"  [{i}] {rel_path}")

    # 2. 用户交互选择
    print("\n" + "-"*60)
    sel_input = input(">>> 请输入要对比的文件序号 (用逗号分隔，如 '0, 1, 3'): ")
    
    try:
        indices = [int(x.strip()) for x in sel_input.split(',')]
        selected_files = [csv_files[i] for i in indices]
    except Exception as e:
        print("❌ 输入格式错误或序号越界，程序退出。")
        return

    # 3. 自定义图例标签
    print("\n" + "-"*60)
    labels = []
    for f in selected_files:
        default_label = os.path.basename(f).replace('.csv', '')
        lbl = input(f">>> 为 '{default_label}' 输入图例名称 (直接回车则使用文件名): ")
        labels.append(lbl if lbl.strip() else default_label)

    print("\n[*] 正在读取数据并生成对比图表...")
    dfs = []
    for f in selected_files:
        dfs.append(pd.read_csv(f))

    # 4. 创建保存目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_target_dir = os.path.join(base_dir, 'result', f"{timestamp}_MultiComparison")
    os.makedirs(new_target_dir, exist_ok=True)
    print(f"[*] 📁 创建图表输出目录: {new_target_dir}")

    # 5. 学术绘图配置
    metrics = {
        'Target_Error(m)': ('Convergence Error Comparison', 'Mean Error (m)'),
        'Min_Distance(m)': ('Minimum Distance Comparison', 'Min Distance (m)'),
        'Avg_Velocity(m/s)': ('Average Velocity Comparison', 'Avg Velocity (m/s)')
    }

    # 使用 matplotlib 内置的优质离散色板 (Tab10) 和不同的线型以区分多条线
    colors = plt.cm.tab10.colors
    linestyles = ['-', '--', '-.', ':']

    for col, (title, ylabel) in metrics.items():
        # 检查所有被选中的 dataframe 是否都包含这个指标
        if all(col in df.columns for df in dfs):
            plt.figure(figsize=(10, 6))
            
            global_max = 0.0
            
            # 绘制每条数据线
            for idx, df in enumerate(dfs):
                color = colors[idx % len(colors)]
                linestyle = linestyles[idx % len(linestyles)]
                linewidth = 2.5 if idx == 0 else 1.8 # 突出显示第一条线（通常是主推的ATO方法）
                
                plt.plot(df['Time(s)'], df[col], linewidth=linewidth, color=color, 
                         linestyle=linestyle, label=labels[idx], alpha=0.9)
                
                # 计算全局最大值用于动态调整Y轴
                if not df[col].empty:
                    global_max = max(global_max, df[col].max())
            
            # 添加物理基准辅助线
            if col == 'Target_Error(m)':
                plt.axhline(y=0.0, color='black', linestyle=':', label='Ideal (0.0m)')
            elif col == 'Min_Distance(m)':
                plt.axhline(y=0.3, color='black', linestyle='-.', linewidth=1.5, label='Safety Limit (0.3m)')
                plt.axhspan(0, 0.3, color='gray', alpha=0.15)
                plt.ylim(bottom=0.25, top=global_max * 1.05)
            elif col == 'Avg_Velocity(m/s)':
                plt.axhline(y=1.0, color='blue', linestyle=':', alpha=0.5, label='Max Velocity Limit')
                plt.ylim(bottom=-0.05, top=global_max * 1.15 if global_max > 1.0 else 1.1)

            # 学术级图表美化
            plt.title(title, fontweight='bold', fontsize=15)
            plt.xlabel('Time $t$ (s)', fontsize=13)
            plt.ylabel(ylabel, fontsize=13)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='best', fontsize=11, frameon=True, shadow=True)
            
            save_name = f"MultiCompare_{col.split('(')[0]}.png"
            plt.tight_layout()
            
            save_path = os.path.join(new_target_dir, save_name)
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"✅ 成功生成图表: {save_name}")
        else:
            print(f"⚠️ 警告: 数据列 '{col}' 在某些选定的 CSV 中缺失，已跳过。")

    print("\n[*] 🎉 所有对比图表均已生成完毕！")
    print(f"[*] 请前往 {new_target_dir} 查看。")

if __name__ == "__main__":
    generate_multi_comparison_plots()