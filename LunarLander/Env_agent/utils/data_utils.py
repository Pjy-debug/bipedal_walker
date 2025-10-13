import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import math


class MyDataset(Dataset):
    def __init__(self, inputs, labels):
        super(MyDataset, self).__init__()
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx],dtype=torch.float32), torch.tensor(self.labels[idx],dtype=torch.float32)

def create_dataset_new(train_data_pos, train_data_neg,data_type):
    np.random.seed(0)
    print(len(train_data_neg),len(train_data_pos))
    if data_type == 'train':
        samples_data_pos = train_data_pos
        samples_data_neg = train_data_neg
    elif data_type == 'val':
        samples_data_pos = random.sample(train_data_pos,10000)
        samples_data_neg = random.sample(train_data_neg,10000)
    elif data_type == 'test':
        samples_data_pos = random.sample(train_data_pos,250)
        samples_data_neg = random.sample(train_data_neg,20000)
    print(f'num of pos is {len(samples_data_pos)},num of neg is {len(samples_data_neg)},ratio is {len(samples_data_neg)/len(samples_data_pos)}')

    train_data = list(samples_data_pos) + list(samples_data_neg)
    inputs = []
    labels = []

    for k, step_data in enumerate(train_data):
        labels.append((step_data['label'],0))
        
        wind_idx = step_data['wind_idx'] - step_data['env_action']
        wind_mag = math.tanh( math.sin(0.02 * wind_idx) + math.sin(math.pi * 0.01 * wind_idx) ) * 10
        
        torque_idx = step_data['torque_idx'] - step_data['env_action']
        torque_mag = math.tanh( math.sin(0.02 * torque_idx)  + math.sin(math.pi * 0.01 * torque_idx)) 
        
    
        input = list(step_data['state'])+ [wind_mag, torque_mag, step_data['env_action']]
        inputs.append(input)
    
    """
    for k, data in enumerate(samples_data_neg):
        labels.append((0,0.0))
        inputs.append(data)
    """
    
    #print(len(inputs))
    # print(inputs)
    # print(inputs[0])
    mydataset = MyDataset(inputs,labels)
    return mydataset



def create_pos_dataset(train_data_pos):
    np.random.seed(0)
    #samples_data_pos = random.sample(train_data_pos,200000)
    samples_data_pos = train_data_pos
    print(f'num of pos is {len(samples_data_pos)}')
   
    train_data = samples_data_pos
    inputs = []
    labels = []

    """
            new_step_data['state'] = step_data['observation']
            new_step_data['pos'] = list(cur_pos)
            new_step_data['terrain'] = cur_terrain
            new_step_data['env_action'] = cur_action
            new_step_data['info'] = other_info
            new_step_data['label'] = 0
    """
    for k, step_data in enumerate(train_data):
        labels.append((step_data['label'],0))
        
        #wind_idx = step_data['wind_idx'] - step_data['env_action']
        wind_idx = step_data['wind_idx']
        wind_mag = math.tanh( math.sin(0.02 * wind_idx) + math.sin(math.pi * 0.01 * wind_idx) ) * 10
        
        #torque_idx = step_data['torque_idx'] - step_data['env_action']
        torque_idx = step_data['torque_idx']
        torque_mag = math.tanh( math.sin(0.02 * torque_idx)  + math.sin(math.pi * 0.01 * torque_idx)) 
        
    
        input = list(step_data['state'])+ [wind_mag, torque_mag, step_data['env_action']]
        inputs.append(input)
        
    mydataset = MyDataset(inputs,labels)
    return mydataset

def create_neg_dataset(train_data_neg):
    
    np.random.seed(0)
    #samples_data_neg = random.sample(samples_data_neg,80000)
    samples_data_neg = train_data_neg
    print(f'num of neg is {len(samples_data_neg)}')
   
    train_data = samples_data_neg
    inputs = []
    labels = []

    """
            new_step_data['state'] = step_data['observation']
            new_step_data['pos'] = list(cur_pos)
            new_step_data['terrain'] = cur_terrain
            new_step_data['env_action'] = cur_action
            new_step_data['info'] = other_info
            new_step_data['label'] = 0
    """
    for k, step_data in enumerate(train_data):
        labels.append((step_data['label'],0))
        
        #wind_idx = step_data['wind_idx'] - step_data['env_action']
        wind_idx = step_data['wind_idx']
        wind_mag = math.tanh( math.sin(0.02 * wind_idx) + math.sin(math.pi * 0.01 * wind_idx) ) * 10
        
        #torque_idx = step_data['torque_idx'] - step_data['env_action']
        torque_idx = step_data['torque_idx']
        torque_mag = math.tanh( math.sin(0.02 * torque_idx)  + math.sin(math.pi * 0.01 * torque_idx)) 
        
    
        input = list(step_data['state'])+ [wind_mag, torque_mag, step_data['env_action']]
        inputs.append(input)
    """
    for k, data in enumerate(train_data):
        labels.append((0,0.0))
        inputs.append(data)
    """
    mydataset = MyDataset(inputs,labels)
    return mydataset

def create_distill_dataset(dataset_path, dataset_type):
    
    np.random.seed(0)

    inputs = []
    labels = []
    # 辅助函数，用于处理单个时间步的数据
    def process_data(samples_data, label_tuple):
        nonlocal inputs, labels
        # 遍历每一个独立的样本
        for step_data in samples_data:
            # 提取特征
            wind_idx = step_data['wind_idx']
            wind_mag = math.tanh(math.sin(0.02 * wind_idx) + math.sin(math.pi * 0.01 * wind_idx)) * 10
            
            torque_idx = step_data['torque_idx']
            torque_mag = math.tanh(math.sin(0.02 * torque_idx) + math.sin(math.pi * 0.01 * torque_idx))
            
            # 组合成单个时间步的特征向量
            cur_input = list(step_data['state']) + [wind_mag, torque_mag, step_data['env_action']]
            
            # 将这个单个时间步的数据添加到总列表
            inputs.append(cur_input)
            labels.append(label_tuple)
    
    if dataset_type == 'total':
        train_data_pos = np.load(dataset_path[0], allow_pickle=True)
        train_data_pos = list(train_data_pos)
        print(f'num of pos dataset is {len(train_data_pos)}')
        train_data_neg = np.load(dataset_path[1], allow_pickle=True)
        train_data_neg = list(train_data_neg)
        print(f'num of neg dataset is {len(train_data_neg)}')
        # 调用辅助函数处理正样本和负样本
        process_data(train_data_pos, (1, 0.0))
        process_data(train_data_neg, (0, 0.0))
        '''for k, step_data in enumerate(train_data_pos):
            labels.append((step_data['label'],0))
        
            #wind_idx = step_data['wind_idx'] - step_data['env_action']
            wind_idx = step_data['wind_idx']
            wind_mag = math.tanh( math.sin(0.02 * wind_idx) + math.sin(math.pi * 0.01 * wind_idx) ) * 10
            
            #torque_idx = step_data['torque_idx'] - step_data['env_action']
            torque_idx = step_data['torque_idx']
            torque_mag = math.tanh( math.sin(0.02 * torque_idx)  + math.sin(math.pi * 0.01 * torque_idx)) 
        
            input = list(step_data['state'])+ [wind_mag, torque_mag, step_data['env_action']]
            inputs.append(input)
        for k, data in enumerate(train_data_neg):
            labels.append((0,0.0))
            inputs.append(data)
        print(f'num of dataset is {len(inputs)}')'''
    else:
        train_data = np.load(dataset_path, allow_pickle=True)
        train_data = list(train_data)
        print(f'num of dataset is {len(train_data)}')
        '''for k, data in enumerate(train_data):
            #print(data)
            if dataset_type == 'pos':
                labels.append((1,0.0))
            elif dataset_type == 'neg':
                labels.append((0,0.0))
            inputs.append(data)'''
        if dataset_type == 'pos':
            process_data(train_data, (1, 0.0))
        elif dataset_type == 'neg':
            process_data(train_data, (0, 0.0))
    print(f'num of total dataset is {len(inputs)}')  
    mydataset = MyDataset(inputs,labels)
    return mydataset
    

 
def create_episode_dataset(train_data_pos, train_data_neg, data_type):
    print("开始处理 create_episode_dataset...") # ⬅️ 新增打印
    np.random.seed(0)
    print(len(train_data_neg),len(train_data_pos))
    
    samples_data_pos = list(train_data_pos)
    samples_data_neg = list(train_data_neg)
    print(f'num of pos is {len(samples_data_pos)},num of neg is {len(samples_data_neg)}')

    train_data = list(samples_data_pos) + list(samples_data_neg)
    inputs = []
    labels = []
    def process_data(samples_data, label_tuple):
        print(f"porcess_data 开始处理 {len(samples_data)} 个episode...") # ⬅️ 新增打印
        nonlocal inputs, labels
        # 遍历每一个 episode
        for episode_data in samples_data:
            # 找到当前 episode 的时间步数
            num_steps = len(episode_data['wind_idx'])
            #print(f'num_steps = {num_steps}')
            
            # 定义切分的窗口大小
            window_size = 11
            if num_steps < window_size:
                print(f"Skipping episode with length {num_steps} which is less than {window_size}")
                continue # 跳过当前循环，进入下一个 episode

            # ➡️ **修改点2：计算并舍弃多余的前面数据**
            # 计算需要从开头舍弃的步数，以使剩余数据长度是 window_size 的整数倍。
            discard_steps = num_steps % window_size
            start_index = discard_steps
            
            # 关键修改: 使用滑动窗口切分
            # 循环从 0 到 num_steps - window_size，生成每一个子序列
            #for i in range(num_steps - window_size + 1):
            for i in range(start_index, num_steps, window_size):
                # 提取当前窗口的子序列数据
                window_inputs = []
                for j in range(window_size):
                    step_index = i + j
                    
                    wind_idx = episode_data['wind_idx'][step_index]
                    wind_mag = math.tanh(math.sin(0.02 * wind_idx) + math.sin(math.pi * 0.01 * wind_idx)) * 10
                    
                    torque_idx = episode_data['torque_idx'][step_index]
                    torque_mag = math.tanh(math.sin(0.02 * torque_idx) + math.sin(math.pi * 0.01 * torque_idx)) 
                    
                    # 组合成单个时间步的特征向量
                    cur_input = list(episode_data['obs'][step_index]) + [wind_mag, torque_mag, episode_data['action'][step_index]]
                    window_inputs.append(cur_input)
                
                # 将这个11步的子序列添加到总输入列表
                inputs.append(window_inputs)
                labels.append(label_tuple)
            #print("所有episode处理完毕，总输入数量:", len(inputs)) # ⬅️ 新增打印
    if data_type == 'total':
        process_data(samples_data_pos, (1, 0))
        process_data(samples_data_neg, (0, 0))
    elif data_type == 'pos':
        process_data(samples_data_pos, (1, 0))
    elif data_type == 'neg':
        process_data(samples_data_neg, (0, 0))
  
    mydataset = MyDataset(inputs,labels)
    print("create_episode_dataset函数执行完毕。") # ⬅️ 新增打印
    return mydataset

''' 
def create_episode_distill_dataset(train_data_pos, train_data_neg, data_type):
    np.random.seed(0)

    inputs = []
    labels = []
    def process_distill_data(samples_data, label_tuple):
        nonlocal inputs, labels
        for k, data in enumerate(samples_data):
            labels.append(label_tuple)
            data = list(data)
            input = []
            for step_data in data:
                wind_idx = step_data['wind_idx'] 
                wind_mag = math.tanh( math.sin(0.02 * wind_idx) + math.sin(math.pi * 0.01 * wind_idx) ) * 10
            
                torque_idx = step_data['torque_idx']
                torque_mag = math.tanh( math.sin(0.02 * torque_idx)  + math.sin(math.pi * 0.01 * torque_idx)) 
            
                cur_input = list(step_data['state'])+ [wind_mag, torque_mag, step_data['env_action']]
                input += cur_input
            inputs.append(input)

    if data_type == 'total':
        process_distill_data(train_data_pos, (1,0))
        for k, data in enumerate(train_data_pos):
            labels.append((1,0))
            data = list(data)
            input = []
            for step_data in data:
                #wind_idx = step_data['wind_idx'] - step_data['env_action']
                wind_idx = step_data['wind_idx'] 
                wind_mag = math.tanh( math.sin(0.02 * wind_idx) + math.sin(math.pi * 0.01 * wind_idx) ) * 10
            
                #torque_idx = step_data['torque_idx'] - step_data['env_action']
                torque_idx = step_data['torque_idx']
                torque_mag = math.tanh( math.sin(0.02 * torque_idx)  + math.sin(math.pi * 0.01 * torque_idx)) 
            
        
                cur_input = list(step_data['state'])+ [wind_mag, torque_mag, step_data['env_action']]
                input += cur_input
            inputs.append(input)
        for k, data in enumerate(train_data_neg):
            labels.append((0,0))
            inputs.append(data)
    elif data_type == 'pos':
        process_distill_data(train_data_pos, (1,0))
    elif data_type == 'neg':
        for k, data in enumerate(train_data_neg):
            labels.append((0,0))
            inputs.append(data)
    
    
  
    mydataset = MyDataset(inputs,labels)
    return mydataset
'''
# === 以下是修改过的函数 ===
def create_episode_distill_dataset(train_data_pos, train_data_neg, data_type):
    inputs, labels = [], []
    def process_data(samples_data, label_tuple):
        print(f"porcess_data 开始处理 {len(samples_data)} 个episode...") # ⬅️ 新增打印
        nonlocal inputs, labels
        # 遍历每一个 episode
        for episode_data in samples_data:
            # 找到当前 episode 的时间步数
            num_steps = len(episode_data['wind_idx'])
            print(f'num_steps = {num_steps}')
            
            # 定义切分的窗口大小
            window_size = 11
            if num_steps < window_size:
                print(f"Skipping episode with length {num_steps} which is less than {window_size}")
                continue # 跳过当前循环，进入下一个 episode

            # ➡️ **修改点2：计算并舍弃多余的前面数据**
            # 计算需要从开头舍弃的步数，以使剩余数据长度是 window_size 的整数倍。
            discard_steps = num_steps % window_size
            start_index = discard_steps
            
            # 关键修改: 使用滑动窗口切分
            # 循环从 0 到 num_steps - window_size，生成每一个子序列
            #for i in range(num_steps - window_size + 1):
            for i in range(start_index, num_steps, window_size):
                # 提取当前窗口的子序列数据
                window_inputs = []
                for j in range(window_size):
                    step_index = i + j
                    
                    wind_idx = episode_data['wind_idx'][step_index]
                    wind_mag = math.tanh(math.sin(0.02 * wind_idx) + math.sin(math.pi * 0.01 * wind_idx)) * 10
                    
                    torque_idx = episode_data['torque_idx'][step_index]
                    torque_mag = math.tanh(math.sin(0.02 * torque_idx) + math.sin(math.pi * 0.01 * torque_idx)) 
                    
                    # 组合成单个时间步的特征向量
                    cur_input = list(episode_data['obs'][step_index]) + [wind_mag, torque_mag, episode_data['action'][step_index]]
                    window_inputs.append(cur_input)
                
                # 将这个11步的子序列添加到总输入列表
                inputs.append(window_inputs)
                labels.append(label_tuple)
            print("所有episode处理完毕，总输入数量:", len(inputs)) # ⬅️ 新增打印
    if data_type == 'pos':
        # 修改：循环遍历正样本数据
        process_data(train_data_pos, (1, 0))
    elif data_type == 'neg':
        # ⬅️ 修改点2：同样在 total 分支中添加对负样本长度的检查
        window_size = 11
        for k, data in enumerate(train_data_neg):
            # 只有当子列表长度为 window_size (11) 时才添加
            if len(data) == window_size:
                labels.append((0,0))
                inputs.append(data)
            else:
                print(f"警告：跳过负样本，其长度为 {len(data)}，不等于 {window_size}")
    elif data_type == 'total':
        process_data(train_data_pos, (1, 0))
        # ⬅️ 修改点2：同样在 total 分支中添加对负样本长度的检查
        window_size = 11
        for k, data in enumerate(train_data_neg):
            # 只有当子列表长度为 window_size (11) 时才添加
            if len(data) == window_size:
                labels.append((0,0))
                inputs.append(data)
            else:
                print(f"警告：跳过负样本，其长度为 {len(data)}，不等于 {window_size}")
    return MyDataset(inputs, labels)