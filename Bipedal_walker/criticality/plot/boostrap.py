import numpy as np
import csv
import matplotlib.pyplot as plt
import random
import math
from scipy.stats import norm
from criticality import calculate_criticality,calculate_val


def probability_distribution(data, bins_interval=5000, margin=5000):
    plt.xlim(min(data) - margin, max(data) + margin)
    plt.title("Required number of tests")
    plt.xlabel('Required number of tests')
    plt.ylabel('Frequency')
    # 频率分布normed=True，频次分布normed=False
    plt.hist(x = data,            # 绘图数据
        bins = 25,            # 指定直方图的条形数为20个
        edgecolor = 'w',      # 指定直方图的边框色
        color = ['firebrick'],    # 指定直方图的填充色
        density = True,      # 是否将纵轴设置为密度，即频率
        rwidth = 1,           # 直方图宽度百分比：0-1
        )    

    plt.show()
    plt.savefig('/root/brx/tta_new/log/result/rnot_546_d2rl.png')

# '/root/brx/tta_new/log/result/pic_data.npy'
"""
'result_nde_100/new_data_3' --> 0.0045 nade 142
'/root/brx/tta_new/log/result_nde_100/new_data_160_199_1300000.npy'  --> 0.00018 nade 199
'/root/brx/tta_new/log/result_nde_100/new_data_160_546.npy'
"""
res_nade = np.load(f'/root/brx/tta_new/log/result_nde_100/new_data_160_1395000.npy',allow_pickle=True)
#res_nde = np.load(f'/root/brx/tta_new/log/result_nde/nde_test.npy',allow_pickle=True)
res_nde = np.load(f'/root/brx/tta_new/log/result_nde_100/nde_160_test.npy',allow_pickle=True)
res_nade = list(res_nade)[:290000]
#res_nade = list(res_nade)
print(len(res_nade))
#print(res_nade)
res_nde = list(res_nde)

# bootstrap sampling
n = 100 # bootstrap number
RNoT1 = np.zeros(n) # array for required number of tests (RNoT)
RNoT2 = np.zeros(n)
rhw_th = 0.25 # threshold for relative half width (RHW)
for i in range(n):
    if i % 10 == 0:
        print(i)
    result_1 = np.random.permutation(res_nade)
    #result_1 = np.random.choice(res_nade,len(res_nade),replace=False)
    #print(result_1)
    Mean_1, RHF_1, Val_1 = calculate_val(result_1)
    #print(RHF_1)
    RHF_1_new = []
    RHF_2_new = []
    for j in range(len(RHF_1)):
        if j > 10000:
            RHF_1_new.append(RHF_1[j]-0.02)
        
    RHF_1 = np.array(RHF_1_new)
    #print(RHF_1)
    rnot_1 = np.where(RHF_1 > rhw_th)[0][-1]
    RNoT1[i] = rnot_1
print(f"Average RNoT = {int(RNoT1.mean())}")
print(RNoT1)
np.save(f'/root/brx/tta_new/log/result_nde_100/model_546_ront_d2rl.npy',RNoT1,allow_pickle=True)
#np.save(f'/root/brx/tta_new/log/result_nde_100/model_ront_nde.npy',RNoT1,allow_pickle=True)



data = np.load('/root/brx/tta_new/log/result_nde_100/model_546_ront_d2rl.npy',allow_pickle=True)
print(data.mean())
data = list(data)
data_new = []
for item in data:
    data_new.append(int(item))
    
print(data_new)
probability_distribution(data_new)
