import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import math
import random
import re
from scipy.stats import norm

# ----------------- 数据处理和绘图函数 -----------------

def calculate_val(the_list):
    Mean=[]
    Relative_half_width=[]
    Var=[]
    var_old=0
    mean_old=0
    for i in range(len(the_list)):
        if math.isnan(the_list[i]) or math.isinf(the_list[i]):
            the_list[i]=0.0
        n=i+1
        mean_new=mean_old+(the_list[i]-mean_old)/n
        Mean.append(mean_new)
        var_new=(n-1)*var_old/n+(n-1)*(the_list[i]-mean_old)**2/(n*n)
        Var.append(1.96*(np.sqrt(var_new/n)))
        Relative_half_width.append(z*(np.sqrt(var_new/n)/(mean_new+1e-30)))
        var_old=var_new
        mean_old=mean_new
    return Mean,Relative_half_width,Var

def load_and_merge_nade_data(base_path='test_results', process_ids=None):
    if process_ids is None:
        print("Error: No process IDs provided. Please specify which processes to merge.")
        return []

    all_nade_crashes = []
    
    for i in process_ids:
        proc_path = os.path.join(base_path, f'process_{i}')
        if not os.path.isdir(proc_path):
            print(f"Warning: Directory not found for process {i}. Skipping.")
            continue
            
        latest_file = None
        latest_epoch = -1
        
        # 寻找最新的 crash_data_d2rl_new_*.npy 文件
        for filename in os.listdir(proc_path):
            if filename.startswith('crash_data_d2rl_new_') and filename.endswith('.npy'):
                match = re.search(r'crash_data_d2rl_new_(\d+)\.npy', filename)
                if match:
                    epoch = int(match.group(1))
                    if epoch > latest_epoch:
                        latest_epoch = epoch
                        latest_file = filename
        
        if latest_file:
            file_path = os.path.join(proc_path, latest_file)
            print(f"Loading NADE data from latest file: {file_path}")
            data = np.load(file_path, allow_pickle=True)
            # 从每个回合的字典中提取 'crash' 值
            crashes_from_data = [d['crash'] for d in data]
            all_nade_crashes.extend(crashes_from_data)
        else:
            print(f"No crash_data_d2rl_new_*.npy files found in {proc_path}. Skipping.")
                
    return all_nade_crashes

def load_nde_data(pkl_file_path, npy_dir):
    all_crashes = []
    
    if os.path.exists(pkl_file_path):
        print(f"Loading NDE data from '{pkl_file_path}'...")
        with open(pkl_file_path, 'rb') as f:
            data_dict = pickle.load(f)
            all_crashes = data_dict['crashes']
    else:
        print(f"'{pkl_file_path}' not found. Loading NDE crash data from individual .npy files in '{npy_dir}'...")
        npy_files = [f for f in os.listdir(npy_dir) if f.startswith('crash_new_450_orin_proc_') and f.endswith('.npy')]
        if npy_files:
            for f_name in npy_files:
                file_path = os.path.join(npy_dir, f_name)
                crashes_data = np.load(file_path)
                all_crashes.extend(crashes_data)
            print(f"Total NDE crash data points loaded: {len(all_crashes)}")
        else:
            print("No NDE data files found. Please run the data collection script first.")

    return all_crashes

def plot_multiple_means(data_tuples, xlabel, ylabel, title, save_path=None):
    font = {'family' : 'serif', 'size' : 15}
    plt.rc('font', **font)
    plt.figure(figsize=(10, 6))
    
    colors = ['#830518', '#058318', '#051883', '#838305', '#188383']
    
    for i, (x_data, y_data, legend_label) in enumerate(data_tuples):
        color = colors[i % len(colors)]
        plt.plot(x_data, y_data, color=color, label=legend_label)

    plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.legend(loc='best')
    plt.title(title, fontsize=25)
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=400)
    
    plt.show()

# ----------------- Main execution part -----------------

if __name__ == '__main__':
    successful_process_ids = [0, 1, 2, 3] 

    # --- 加载 NADE 失败率数据 ---
    nade_crashes = load_and_merge_nade_data(process_ids=successful_process_ids)
    
    if nade_crashes:
        mean_nade, _, _ = calculate_val(nade_crashes)
    else:
        mean_nade = []
        print("NADE data not found or loaded incorrectly.")

    # --- 加载 NDE 失败率数据 ---
    pkl_file_path = 'ndereults/ordinary_multiprocess.pkl'
    npy_dir = 'ndereults'
    crashes_nde = load_nde_data(pkl_file_path, npy_dir)

    if crashes_nde:
        mean_nde, _, _ = calculate_val(crashes_nde)
    else:
        mean_nde = []
        print("NDE crash data not found or loaded incorrectly.")

    # --- 绘图 ---
    if mean_nde or mean_nade:
        mean_data_tuples = []
        if mean_nde:
            mean_data_tuples.append((range(len(mean_nde)), mean_nde, 'NDE (Ordinary Method)'))
        if mean_nade:
            mean_data_tuples.append((range(len(mean_nade)), mean_nade, 'NADE'))
        
        plot_multiple_means(
            mean_data_tuples,
            'Number of Episodes',
            'Failure rate',
            'Failure Rate Comparison',
            save_path='data/combined/failure_rate_comparison.png'
        )