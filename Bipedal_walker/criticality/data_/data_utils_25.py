import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, inputs, labels):
        super(MyDataset, self).__init__()
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx],dtype=torch.float32), torch.tensor(self.labels[idx],dtype=torch.float32)

def create_dataset(dataset_path_pos, dataset_path_neg,ratio, data_type):
    train_data_pos = np.load(dataset_path_pos, allow_pickle=True)
    train_data_neg = np.load(dataset_path_neg, allow_pickle=True)
    train_data_neg = list(train_data_neg)
    train_data_pos = list(train_data_pos)
    #print(1)
    print(len(train_data_neg),len(train_data_pos))
    np.random.seed(0)
    if ratio == 0:
        if data_type == 'train':
            samples_data_pos = train_data_pos
            samples_data_neg = train_data_neg
            #samples_data_pos = random.sample(train_data_pos,400000)
            # samples_data_neg = train_data_neg[:400000]
           # samples_data_neg = random.sample(train_data_neg,len(samples_data_pos))
        
        elif data_type == 'val':
            #samples_data_pos = random.sample(train_data_pos,10000)
            #samples_data_neg = random.sample(train_data_neg,10000)
            samples_data_pos = train_data_pos
            samples_data_neg = train_data_neg
        elif data_type == 'test':
            #samples_data_pos = random.sample(train_data_pos,1000)
            #samples_data_neg = random.sample(train_data_neg,80000)
            #samples_data_pos = train_data_pos
            #samples_data_neg = train_data_neg
            pass
        #else must be added in python
        else:
            pass
        print(f'num of pos is {len(samples_data_pos)},num of neg is {len(samples_data_neg)},ratio is {len(samples_data_neg)/len(samples_data_pos)}')
    else:
        if data_type == 'train':
            samples_data_pos = train_data_pos
            samples_data_neg = random.sample(train_data_neg,int(len(train_data_neg)*ratio))
        else:
            samples_data_pos = random.sample(train_data_pos,int(len(train_data_neg)/ratio))
            samples_data_neg = train_data_neg
        print(f'num of pos is {len(samples_data_pos)},num of neg is {len(samples_data_neg)},ratio is {len(samples_data_neg)/len(samples_data_pos)}')
    train_data = samples_data_pos + samples_data_neg
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
        labels.append((step_data['label'],step_data['fall_dist']))
        terrain = []
        for pre_terrain in step_data['pre_terrain']:
            for item in pre_terrain:
                terrain += list(item)
        for item in step_data['terrain']:
            terrain += list(item)
        # generate X_k
        input = list(step_data['state'])+ [step_data['env_action']]
        #input = list(step_data['state']) + list(step_data['pos']) + terrain + [step_data['env_action']]
        #print(len(input))
        inputs.append(input)

    #print(len(inputs))
    # print(inputs)
    # print(inputs[0])
    mydataset = MyDataset(inputs,labels)
    return mydataset

"""
dataset_path_pos = '/root/autodl-tmp/data/dataset/train_dataset_pos.npy'
dataset_path_neg = '/root/autodl-tmp/data/dataset/train_dataset_neg.npy'
mydataset = create_dataset(dataset_path_pos, dataset_path_neg, 0, 'train')
# print(len(mydataset),mydataset)
"""

def create_dataset_new(train_data_pos, train_data_neg, ratio, data_type):
    np.random.seed(0)
    print(len(train_data_neg),len(train_data_pos))
    if ratio == 0:
        if data_type == 'train':
            samples_data_pos = train_data_pos
            samples_data_neg = train_data_neg
            #samples_data_pos = train_data_pos[:400000]
            #samples_data_pos = random.sample(train_data_pos,len(train_data_neg))
            #samples_data_neg = train_data_neg[400000:800000]
            #samples_data_neg = random.sample(train_data_neg,600000)
        elif data_type == 'val':
            samples_data_pos = random.sample(train_data_pos,10000)
            samples_data_neg = random.sample(train_data_neg,10000)
        elif data_type == 'test':
            samples_data_pos = random.sample(train_data_pos,250)
            samples_data_neg = random.sample(train_data_neg,20000)
        print(f'num of pos is {len(samples_data_pos)},num of neg is {len(samples_data_neg)},ratio is {len(samples_data_neg)/len(samples_data_pos)}')
    else:
        if data_type == 'train':
            samples_data_pos = train_data_pos
            samples_data_neg = random.sample(train_data_neg,int(len(train_data_neg)*ratio))
        else:
            samples_data_pos = random.sample(train_data_pos,int(len(train_data_neg)/ratio))
            samples_data_neg = train_data_neg
        print(f'num of pos is {len(samples_data_pos)},num of neg is {len(samples_data_neg)},ratio is {len(samples_data_neg)/len(samples_data_pos)}')
    train_data = samples_data_pos + samples_data_neg
    inputs = []
    labels = []

    for k, step_data in enumerate(train_data):
        labels.append((step_data['label'],step_data['fall_dist']))
        terrain = []
        for pre_terrain in step_data['pre_terrain']:
            for item in pre_terrain:
                terrain += list(item)
        for item in step_data['terrain']:
            terrain += list(item)
        input = list(step_data['state']) + list(step_data['pos']) + terrain + [step_data['env_action']]
        #print(len(input))
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

def create_dataset_new_2(dataset_path_pos, dataset_paths_neg):
    train_data_pos = np.load(dataset_path_pos, allow_pickle=True)
    train_data_pos = list(train_data_pos)
    samples_data_pos = random.sample(train_data_pos,100000)
    
    samples_data_neg = []
    for dataset_path_neg in dataset_paths_neg:
        print(dataset_path_neg)
        train_data_neg = list(np.load(dataset_path_neg, allow_pickle=True))
        #sample_data_neg = random.sample(train_data_neg,int(len(samples_data_pos)/len(dataset_paths_neg)))
        samples_data_neg += train_data_neg
    
    print(f'num of pos is {len(samples_data_pos)},num of neg is {len(samples_data_neg)}')
   
    #train_data = samples_data_pos + samples_data_neg
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
        labels.append((step_data['label'],step_data['fall_dist']))
        terrain = []
        for pre_terrain in step_data['pre_terrain']:
            for item in pre_terrain:
                terrain += list(item)
        for item in step_data['terrain']:
            terrain += list(item)
        input = list(step_data['state']) + list(step_data['pos']) + terrain + [step_data['env_action']]
        #print(len(input))
        inputs.append(input)

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
        labels.append((step_data['label'],step_data['fall_dist']))
        terrain = []
        for pre_terrain in step_data['pre_terrain']:
            for item in pre_terrain:
                terrain += list(item)
        for item in step_data['terrain']:
            terrain += list(item)
        # ȥ����pos
        input = list(step_data['state']) + [step_data['env_action']]
        #print(len(input))
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
        labels.append((step_data['label'],step_data['fall_dist']))
        terrain = []
        for pre_terrain in step_data['pre_terrain']:
            for item in pre_terrain:
                terrain += list(item)
        for item in step_data['terrain']:
            terrain += list(item)
        # ȥ����pos
        input = list(step_data['state']) + [step_data['env_action']]
        #print(len(input))
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
    
    if dataset_type == 'total':
        train_data_pos = np.load(dataset_path[0], allow_pickle=True)
        train_data_pos = list(train_data_pos)
        print(f'num of pos dataset is {len(train_data_pos)}')
        train_data_neg = np.load(dataset_path[1], allow_pickle=True)
        train_data_neg = list(train_data_neg)
        print(f'num of neg dataset is {len(train_data_neg)}')
        for k, data in enumerate(train_data_pos):
            labels.append((1,0.0))
            inputs.append(data)
        for k, data in enumerate(train_data_neg):
            labels.append((0,0.0))
            inputs.append(data)
        print(f'num of dataset is {len(inputs)}')
    else:
        train_data = np.load(dataset_path, allow_pickle=True)
        train_data = list(train_data)
        print(f'num of dataset is {len(train_data)}')
        for k, data in enumerate(train_data):
            #print(data)
            if dataset_type == 'pos':
                labels.append((1,0.0))
            elif dataset_type == 'neg':
                labels.append((0,0.0))
            inputs.append(data)
        
    mydataset = MyDataset(inputs,labels)
    return mydataset