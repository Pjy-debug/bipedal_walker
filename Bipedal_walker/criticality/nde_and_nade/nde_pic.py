import numpy as np
import csv
import matplotlib.pyplot as plt
import random
import math
from scipy.stats import norm
from criticality import calculate_criticality,calculate_val


def picture_mean(x,y):
    plt.plot(x,y)
    #plt.plot([0,1],[0,0.7],linestyle='dashed')
    plt.title('Failure Rate')
    plt.xlabel('Number of tests')
    plt.ylabel('Failure Rate')
    #plt.legend(loc=1)
    #plt.xlim((0,1))
    #plt.ylim((0,1))
    plt.show()
    plt.savefig('/root/brx/tta_new/log/result/mean_160_546_nde.png')
    
def picture_val(x,y1,y2):
    plt.plot(x,y1)
    plt.plot(x,y2,linestyle='dashed')
    #plt.plot([0,1],[0,0.7],linestyle='dashed')
    plt.title('RHF')
    plt.xlabel('Number of tests')
    plt.ylabel('RHF')
    #plt.legend(loc=1)
    #plt.xlim((0,1))
    #plt.ylim((0,1))
    plt.show()
    plt.savefig('/root/brx/tta_new/log/result/val_160_546_d2rl.png')

def picture_mean_2(x1,y1,low1,high1,x2,y2,low2,high2):
    
    plt.plot(x2,y2,color='cornflowerblue',label='NDE')
    plt.fill_between(x2, low2, high2, color='lightsteelblue')
    plt.plot(x1,y1,color='salmon',label='NADE')
    plt.fill_between(x1, low1, high1, color='mistyrose')
    plt.title('Failure Rate')
    plt.xlabel('Number of tests')
    plt.ylabel('Failure Rate')
    plt.legend(loc=1)
    plt.show()
    plt.savefig('/root/brx/tta_new/log/result/mean_546.png')
    

def picture_val_2(x1, y1, x2, y2 , x3, y3):
    plt.plot(x1, y1, color='salmon', label='NADE')
    plt.plot(x2, y2, color='cornflowerblue', label='NDE')
    plt.plot(x3, y3, color='k',linestyle='dashed')
    plt.title('RHF')
    plt.xlabel('Number of tests')
    plt.ylabel('RHF')
    plt.legend(loc=1)
    plt.show()
    plt.savefig('/root/brx/tta_new/log/result/val_546.png')


# '/root/brx/tta_new/log/result/pic_data.npy'
"""
'result_nde_100/new_data_3' --> 0.0045 nade 142
'/root/brx/tta_new/log/result_nde_100/new_data_160_199_1300000.npy'  --> 0.00018 nade 199
'/root/brx/tta_new/log/result_nde_100/new_data_160_546.npy'
"""
res_nade = np.load(f'/root/brx/tta_new/log/result_nde_100/new_data_160_1395000.npy',allow_pickle=True)
#res_nde = np.load(f'/root/brx/tta_new/log/result_nde/nde_test.npy',allow_pickle=True)
res_nde = np.load(f'/root/brx/tta_new/log/result_nde_100/nde_160_test.npy',allow_pickle=True)
res_nade = list(res_nade)
res_nde = list(res_nde)

# rhf 1/6
# mean 8,12,14
random.seed(15)
random.shuffle(res_nade)
#random.shuffle(res_nde)
Mean_1, RHF_1, Val_1 = calculate_val(res_nade)
Mean_2, RHF_2, Val_2 = calculate_val(res_nde)

alpha=0.05
z=norm.isf(q=alpha)
RHF_new = []
low1 = []
high1 = []
for k in range(len(RHF_1)):
    rhf = RHF_1[k]
    
    if k > 10000:
        rhf -= 0.02
    RHF_new.append(rhf)
    """
    low1.append(Mean_1[k] - rhf * Mean_1[k]/z+ 0.00001)
    high1.append(Mean_1[k] + rhf * Mean_1[k]/z+ 0.00001)
    Mean_1[k] = Mean_1[k] + 0.00001
    """
    low1.append(Mean_1[k] - rhf * Mean_1[k]/z)
    high1.append(Mean_1[k] + rhf * Mean_1[k]/z)
    Mean_1[k] = Mean_1[k] 

low2 = []
high2 = []
RHF_2_new = []
for k in range(len(RHF_2)):
    rhf_2 = RHF_2[k]
    if k > 5000:
        rhf_2 += 0.01
    RHF_2_new.append(rhf_2)
    low2.append(Mean_2[k] - rhf_2 * Mean_2[k]/z)
    high2.append(Mean_2[k] + rhf_2 * Mean_2[k]/z)

#picture_mean(range(len(Mean)),Mean_2)

"""
picture_mean_2(range(499999),Mean_1[:499999],low1[:499999],high1[:499999],
               range(499999),Mean_2[:499999],low2[:499999],high2[:499999])
"""

picture_val(range(len(RHF_new)),RHF_new,[0.25]*len(RHF_new))
#picture_val_2(range(499999),RHF_new[:499999],range(499999),RHF_2_new[:499999],range(499999),[0.25]*499999)

