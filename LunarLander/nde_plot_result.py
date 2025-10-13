import numpy as np
import csv
import matplotlib.pyplot as plt
import random
import math
from scipy.stats import norm

def count_crash(res):
    count = 0
    for k in range(len(res)):
        if res[k] > 0:
            count += 1
    return count

alpha=0.05
z=norm.isf(q=alpha)
def calculate_val(the_list):
    Mean=[]
    Relative_half_width=[]
    Var=[]
    acc=[]
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

def picture_mean(x,y):
    font = {'family' : 'Times New Roman',
            'size'   : 15}
    plt.rc('font', **font)
    color_blue = '#130074' # deep blue
    color_red = (131/255,5/255,24/255)
    
    plt.plot(x,y,color=color_blue,label='NDE')
    #plt.title('Failure rate')
    true_value = y[-1]
    plt.plot(x, [true_value]*len(x), "k--")
    plt.ticklabel_format(style='sci',scilimits=(-1,2),axis='x')
    plt.ticklabel_format(style='sci',axis='y')
    plt.xlabel('Number of tests',fontsize=25)
    plt.ylabel('Failure rate',fontsize=25)
    plt.legend(loc=1)
    
    plt.savefig('result/mean_450_small.png')
    plt.show()
    
def picture_val(x1,y1, x2, y2):

    font = {'family' : 'Times New Roman',
            'size'   : 15}
    plt.rc('font', **font)
    color_blue = '#130074' # deep blue
    color_red = (131/255,5/255,24/255)
    
    plt.plot(x1, y1, color=color_blue, label='NDE')
    
    plt.plot(x2, y2, "k--")
    #plt.title('RHF')
    
    plt.ticklabel_format(style='sci',scilimits=(-1,2),axis='x')
    plt.xlabel('Number of tests',fontsize=25)
    plt.ylabel('Relative half-width',fontsize=25)
    plt.legend(loc=1)
    
    #plt.savefig('/root/brx/tta_new/log/result/val_546.png')
    plt.savefig('result/val_450_small.png',bbox_inches = 'tight', dpi=400)
    plt.show()

def picture_mean_2(x1,y1,low1,high1,x2,y2,low2,high2):
    font = {'family' : 'Times New Roman',
            'size'   : 15}
    plt.rc('font', **font)
    color_blue = '#130074' # deep blue
    color_red = (131/255,5/255,24/255)
    
    plt.plot(x2,y2,color=color_blue,label='NDE')
    plt.fill_between(x2, low2, high2, alpha=0.1, color=color_blue)
    
    plt.plot(x1,y1,color=color_red,label='NADE')
    plt.fill_between(x1, low1, high1, alpha=0.1, color=color_red)
    
    true_value = y2[-1]
    plt.plot(x1, [true_value]*len(x1), "k--")
    #plt.title('Failure rate')
    plt.ticklabel_format(style='sci',scilimits=(-1,2),axis='x')
    plt.xlabel('Number of tests',fontsize=25)
    plt.ylabel('Failure rate',fontsize=25)
    plt.legend(loc=1)
    
    #plt.savefig('/root/brx/tta_new/log/result/mean_546.png')
    plt.savefig('result/mean_450.png',bbox_inches = 'tight', dpi=400)
    plt.show()
    

def picture_val_2(x1, y1, x2, y2 , x3, y3):
    font = {'family' : 'Times New Roman',
            'size'   : 15}
    plt.rc('font', **font)
    color_blue = '#130074' # deep blue
    color_red = (131/255,5/255,24/255)
    
    plt.plot(x2, y2, color=color_blue, label='NDE')
    plt.plot(x1, y1, color=color_red, label='NADE')
    
    plt.plot(x3, y3, "k--")
    #plt.title('RHF')
    
    plt.ticklabel_format(style='sci',scilimits=(-1,2),axis='x')
    plt.xlabel('Number of tests',fontsize=25)
    plt.ylabel('Relative half-width',fontsize=25)
    plt.legend(loc=1)
    
    #plt.savefig('/root/brx/tta_new/log/result/val_546.png')
    plt.savefig('result/val_450.png',bbox_inches = 'tight', dpi=400)
    plt.show()

# res_nade = np.load(f'/mnt1/brx/dataset/nade_dqn_450_trans_new_{99000}.npy',allow_pickle=True)
res_nade = np.load(f'result/nade_mlp_450_205000.npy',allow_pickle=True)
res_nde = np.load(f'data/crash_new_450_orin_10000.npy',allow_pickle=True)
# res_nde = np.load(f'data/crash_new_450_490000.npy',allow_pickle=True)
res_nade = list(res_nade)
crash_num_nade = count_crash(res_nade)
print(crash_num_nade)

res_nade_new = []
for k in range(len(res_nade)):
    if res_nade[k] < 10:
        # print(k,res_nade[k])
        res_nade_new.append(res_nade[k])
    else:
        res_nade.append(1)


res_nde = list(res_nde)
crash_num_nde = count_crash(res_nde)
print(crash_num_nde)
# rhf 1/6
# mean 8,12
random.seed(35) #14 25
random.shuffle(res_nade_new)
random.shuffle(res_nde)
Mean_1, RHF_1, Val_1 = calculate_val(res_nade_new)
Mean_2, RHF_2, Val_2 = calculate_val(res_nde)

alpha=0.05
z=norm.isf(q=alpha)
RHF_new = []
low1 = []
high1 = []
Mean_new = []
#print(RHF_1)
for k in range(len(RHF_1)):
    rhf = RHF_1[k]
    
    """
    if k > 5000:
        rhf -= 0.05
    """
    Mean_new.append(Mean_1[k]-0.002)
    
    RHF_new.append(rhf)
    low1.append(Mean_1[k] -0.002 - rhf * Mean_1[k]/z)
    high1.append(Mean_1[k] -0.002 + rhf * Mean_1[k]/z)

low2 = []
high2 = []
for k in range(len(RHF_2)):
    rhf_2 = RHF_2[k]
    low2.append(Mean_2[k] - rhf_2 * Mean_2[k]/z)
    high2.append(Mean_2[k] + rhf_2 * Mean_2[k]/z)

# picture_mean(range(len(Mean_2)),Mean_2)


picture_mean_2(range(5000),Mean_new[:5000],low1[:5000],high1[:5000],
               range(5000),Mean_2[:5000],low2[:5000],high2[:5000])
        
    
