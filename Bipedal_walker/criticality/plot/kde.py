import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

def kde(Y0_Ag,Y1_Ag):
    sns.set_style("darkgrid"
                  #, {"axes.facecolor": "pink"}#修改背景色
                 )#设置背景
    font1 = {'family' : 'Times New Roman'
             ,'weight' : 'normal'
            }#定义一个字体
    font2 = {'family' : 'Times New Roman'
             ,'weight' : 'normal'
            ,'size' : 18}#定义一个字体
    # ax.set_xlim(0,50)
    # ax.set_ylim(0,0.1)#设置X，Y轴范围
    plt.figure(figsize=(12,9))#设置画布大小
    ax = sns.kdeplot(Y0_Ag, color="Red", shade=True)
    ax = sns.kdeplot(Y1_Ag, color="Blue", shade=True)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]#设置坐标刻度字体类型
    plt.tick_params(labelsize=21)#同时设置X，Y轴刻度字体大小

    ax.set_xlabel('Train',fontsize=25,fontproperties=font1)
    ax.set_ylabel("Frequency",fontsize=25,fontproperties=font1)#设置X，Y轴Label字体类型
    ax = ax.legend(["0","1"],prop=font2#设置legend字体类型
             #,loc='best',edgecolor='blue'#修改legend位置和框线颜色
                 #,frameon=False #设置legend无框线
                  )
    ax = plt.gca()
    ax.spines['right'].set_color('grey')
    ax.spines['left'].set_color('grey')
    ax.spines['top'].set_color('grey')
    ax.spines['bottom'].set_color('grey')#添加灰色边框线
    #plt.savefig('G_Age_KDE.tiff',dpi=600)#保存图片，可以修改后缀更改格式
    plt.savefig('/root/brx/tta_new/tta/criticality/new_log/result/kde.png')
    plt.show()

dataset_path_pos ='/root/autodl-tmp/data/dataset/val_dataset_pos.npy'
dataset_path_neg ='/root/autodl-tmp/data/dataset/val_dataset_neg.npy'
train_pos = np.load(dataset_path_pos, allow_pickle=True)
train_pos = list(train_pos)
train_neg = np.load(dataset_path_neg, allow_pickle=True)
train_neg = list(train_neg)

print('successfully loading data...')

train_data = train_pos + train_neg

inputs = []
for k, step_data in enumerate(train_data):
    #labels.append((step_data['label'],step_data['fall_dist']))
    #labels.append(step_data['label'])
    terrain = []
    for pre_terrain in step_data['pre_terrain']:
        for item in pre_terrain:
            terrain += list(item)
    for item in step_data['terrain']:
        terrain += list(item)
    input = list(step_data['state']) + list(step_data['pos']) + terrain + [step_data['env_action']]
    #input = list(step_data['state']) + terrain + [step_data['env_action']]
    #print(len(input))
    inputs.append(input)

inputs_pos = inputs[:len(train_pos)]
inputs_neg = inputs[len(train_pos):]
kde(inputs_neg, inputs_pos)