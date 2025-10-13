import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import manifold
import numpy as np
import random


def visual(feat):
    # t-SNE的最终结果的降维与可视化 2/3
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)

    x_ts = ts.fit_transform(feat)

    print(x_ts.shape)  # [num, 2]

    x_min, x_max = x_ts.min(0), x_ts.max(0)

    x_final = (x_ts - x_min) / (x_max - x_min)

    return x_final  # [num,2]


# 设置散点形状
maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
# 设置散点颜色
colors = ['blue','r','#e38c7a', '#656667', '#99a4bc', 'cyan', 'hotpink', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
          ]
# 图例名称
Label_Com = ['a', 'b', 'c', 'd']
# 设置字体格式
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 32,
         }


def plotlabels(S_lowDWeights, Trure_labels, name):
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    print(S_data)
    print(S_data.shape)  # [num, 3]

    for index in range(2):  # 假设总共有两个类别，类别的表示为0,1
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, s=5, marker=maker[index], c=colors[index], edgecolors=colors[index], alpha=0.65,label=f'class {index}')

        #plt.xticks([])  # 去掉横坐标值
        #plt.yticks([])  # 去掉纵坐标值

    plt.title(name, fontsize=32, fontweight='normal', pad=20)
    plt.legend()
    plt.savefig('/root/brx/tta_new/tta/criticality/new_log/result/train_val_feat_668.png')
    plt.show()

print('loading data....')
dataset_path_pos = '/root/autodl-tmp/data/dataset/train_dataset_pos.npy'
train_data_pos = np.load(dataset_path_pos, allow_pickle=True)
train_data_pos = list(train_data_pos)
#samples_data_pos = train_data_pos[:400000]
samples_data_pos = random.sample(train_data_pos,1000)

#dataset_path_neg = ['/root/brx/tta_new/tta/criticality/new_log/result/data_neg_total.npy']
dataset_path_neg ='/root/autodl-tmp/data/dataset/train_dataset_neg.npy' 
#dataset_path_neg = '/root/autodl-tmp/data/dataset/train_dataset_neg_24.npy' 
train_data_neg = np.load(dataset_path_neg, allow_pickle=True)
train_data_neg = list(train_data_neg)
samples_data_neg = random.sample(train_data_neg,80000)


print('successfully loading data...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 1345
model = torch.load('new_log/model_new/mlp.ep668')
model.eval()
model.to(device)

train_data = samples_data_pos + samples_data_neg
inputs = []
labels = []
feats = []

for k, step_data in enumerate(train_data):
    #labels.append((step_data['label'],step_data['fall_dist']))
    labels.append(step_data['label'])
    terrain = []
    
    for pre_terrain in step_data['pre_terrain']:
        for item in pre_terrain:
            terrain += list(item)
    
    for item in step_data['terrain']:
        terrain += list(item)
    input = list(step_data['state']) + list(step_data['pos']) + terrain + [step_data['env_action']]
    
    
    input = torch.reshape(torch.Tensor(input).to(device),(1,-1))
    output, feat,_,_ = model(input,input,0.5)
    feat = list(feat.squeeze(0).to('cpu').detach().numpy())
    feats.append(feat)
    
    
    #input =  terrain 
    #print(len(input))
    inputs.append(input)
    
    
"""
dataset_path_neg ='/root/brx/tta_new/tta/criticality/new_log/result/data_neg_total.npy'
train_neg = np.load(dataset_path_neg, allow_pickle=True)
train_neg = list(train_neg)
train_data_neg = []
for items in train_neg:
    for item in items:
        train_data_neg.append(item)
samples_data_neg = random.sample(train_data_neg,10000)

print('successfully loading data...')

train_data = samples_data_pos 
inputs = []
labels = []

for k, step_data in enumerate(train_data):
    #labels.append((step_data['label'],step_data['fall_dist']))
    labels.append(step_data['label'])
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
for k, data in enumerate(samples_data_neg):
    inputs.append(data)
    labels.append(0)
"""

feature = np.array(feats)  # 128个特征，每个特征的维度为1024
print(feature.shape)
label_test = np.array(labels)
print(label_test.shape)

fig = plt.figure(figsize=(8, 8))

plotlabels(visual(feature), label_test, 'tsne')
