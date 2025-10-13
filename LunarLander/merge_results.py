import numpy as np
import os
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import re

# 复制原始代码中的 calculate_val 函数
alpha = 0.05
z = norm.isf(q=alpha)
def calculate_val(the_list):
    Mean = []
    Relative_half_width = []
    Var = []
    var_old = 0
    mean_old = 0
    for i in range(len(the_list)):
        if math.isnan(the_list[i]) or math.isinf(the_list[i]):
            the_list[i] = 0.0
        n = i + 1
        mean_new = mean_old + (the_list[i] - mean_old) / n
        Mean.append(mean_new)
        var_new = (n - 1) * var_old / n + (n - 1) * (the_list[i] - mean_old) ** 2 / (n * n)
        Var.append(1.96 * (np.sqrt(var_new / n)))
        Relative_half_width.append(z * (np.sqrt(var_new / n) / (mean_new + 1e-30)))
        var_old = var_new
        mean_old = mean_new
    return Mean, Relative_half_width, Var

def merge_and_analyze_results(base_path='test_results_new_weight', process_ids=None):
    if process_ids is None:
        print("Error: No process IDs provided. Please specify which processes to merge.")
        return

    all_crashes = []
    
    # 遍历你指定的进程 ID 列表
    for i in process_ids:
        proc_path = os.path.join(base_path, f'process_{i}')
        if not os.path.isdir(proc_path):
            print(f"Warning: Directory not found for process {i}. Skipping.")
            continue
            
        latest_file = None
        latest_epoch = -1
        
        # 查找最新的 .npy 文件
        for filename in os.listdir(proc_path):
            # 更改这里的文件名匹配模式
            if filename.startswith('d2rl_new_') and filename.endswith('.npy'):
                # 使用正则表达式提取 epoch 编号
                match = re.search(r'd2rl_new_(\d+)\.npy', filename)
                if match:
                    epoch = int(match.group(1))
                    if epoch > latest_epoch:
                        latest_epoch = epoch
                        latest_file = filename
        
        if latest_file:
            file_path = os.path.join(proc_path, latest_file)
            print(f"Loading data from latest file: {file_path}")
            data = np.load(file_path, allow_pickle=True)
            all_crashes.extend(data)
        else:
            print(f"No d2rl_new_*.npy files found in {proc_path}. Skipping.")
                
    if not all_crashes:
        print("No crash data found from the specified processes.")
        return

    print(f"Total data points after merging: {len(all_crashes)}")

    # 对合并后的数据计算 RHF
    Mean, RHF, Var = calculate_val(all_crashes)
    
    # 新增的逻辑：使用线性插值找到 RHF = 0.3 时的横坐标
    rhf_target = 0.3
    interpolated_x = None

    for i in range(1, len(RHF)):
        if RHF[i-1] >= rhf_target and RHF[i] < rhf_target:
            x1 = i - 1
            y1 = RHF[i-1]
            x2 = i
            y2 = RHF[i]
            
            interpolated_x = x1 + (rhf_target - y1) * (x2 - x1) / (y2 - y1)
            print(f"\nRelative Half-Width (RHF) reaches {rhf_target} at approximately test number: {interpolated_x:.2f}")
            break
            
    if interpolated_x is None:
        if len(RHF) > 0 and RHF[-1] > rhf_target:
             print(f"\nRelative Half-Width (RHF) has not yet reached {rhf_target} in the provided data.")
        else:
            print("\nRelative Half-Width (RHF) has already passed 0.3 at the start of the data.")

    # 绘制 RHF 曲线
    font = {'family': 'serif',
            'size': 15}
    plt.rc('font', **font)
    
    plt.figure(figsize=(12, 6))
    plt.plot(RHF)
    
    plt.xlabel('Number of tests', fontsize=20)
    plt.ylabel('Relative Half-Width (RHF)', fontsize=20)
    plt.title('RHF Convergence Curve (Merged Data)', fontsize=20)
    
    plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
    
    plt.grid(True)
    
    plt.savefig('merged_RHF_convergence.png', bbox_inches='tight', dpi=400)
    plt.show()
    print("RHF convergence plot saved as 'merged_RHF_convergence.png'")
    
    print(f"Final RHF value: {RHF[-1]}")
    
if __name__ == '__main__':
    successful_process_ids = [0,1,2,3,4]
    merge_and_analyze_results(process_ids=successful_process_ids)