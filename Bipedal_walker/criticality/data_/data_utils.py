'''*************************************************************************
[file name]         data_utils.py
[description]       数据管理有关的函数
[author]            brx, 2024
[changelog]         （若修改过则必需注明）
*************************************************************************'''
import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import logging
import tqdm

'''*************************************************************************
[class name]        MyDataset(torch.utils.data.Dataset)
[description]       管理数据, 继承于Dataset, 为DataLoader提供数据接口
[params]            list | inputs: 输入数据
                    list | labels: 标签数据
[author]            brx, 2024
[changelog]         （若修改过则必需注明）
*************************************************************************'''
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
    '''*************************************************************************
    [function name]         create_dataset
    [description]           根据数据集类型train val test来创建数据集
    [input]                 dataset_path_pos: 正数据集路径 
                            dataset_path_neg: 负数据集路径
                            ratio: neg/pos比例 if ratio == 0, ratio is not used
                            data_type: "total" "pos" "neg" 希望返回的数据用途
    [output]                MyDataset | mydataset: mydataset.input = list(step_data['state']) + 
                                list(step_data['pos'])  + terrain + [step_data['env_action']]
    [author]                brx, 2024
    [changelog]             部分代码目的还不懂
    *************************************************************************'''
    train_data_pos = np.load(dataset_path_pos, allow_pickle=True)
    train_data_neg = np.load(dataset_path_neg, allow_pickle=True)
    train_data_neg = list(train_data_neg)
    train_data_pos = list(train_data_pos)
    #print(1)
    print(len(train_data_neg),len(train_data_pos))
    np.random.seed(0)
    if ratio == 0:
        samples_data_pos = None
        samples_data_neg = None
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
            samples_data_pos = train_data_pos
            samples_data_neg = train_data_neg

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
        input = list(step_data['state'])+ list(step_data['pos'])  + terrain + [step_data['env_action']] # 24+2+80+1=107
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
    '''*************************************************************************
    [function name]         create_dataset_new
    [description]           
    [input]                
    [output]                MyDataset | mydataset: torch.utils.data.Dataset
    [author]                brx, 2024
    [changelog]             （若修改过则必需注明）
    *************************************************************************'''
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
            print('sampling...')
            samples_data_pos = list(train_data_pos) # 10000
            samples_data_neg = random.sample(list(train_data_neg),500) # 10000
        elif data_type == 'test':
            # samples_data_pos = random.sample(train_data_pos,500)
            # samples_data_neg = random.sample(train_data_neg,40000)
            samples_data_pos = list(train_data_pos)
            samples_data_neg = list(train_data_neg)
        print(f'num of pos is {len(samples_data_pos)},num of neg is {len(samples_data_neg)},ratio is {len(samples_data_neg)/len(samples_data_pos)}')
    else:
        if data_type == 'train':
            samples_data_pos = train_data_pos
            samples_data_neg = random.sample(train_data_neg,int(len(train_data_neg)*ratio))
        else:
            samples_data_pos = random.sample(train_data_pos,int(len(train_data_neg)/ratio))
            samples_data_neg = train_data_neg
        print(f'num of pos is {len(samples_data_pos)},num of neg is {len(samples_data_neg)},ratio is {len(samples_data_neg)/len(samples_data_pos)}')
    
    print("num of pos is {}, num of neg is {}", len(samples_data_pos), len(samples_data_neg))
    train_data = samples_data_pos + samples_data_neg
    inputs = []
    labels = []

    for step_data in tqdm.tqdm(train_data):
        labels.append((step_data['label'],step_data['fall_dist']))
        terrain = []
        for pre_terrain in step_data['pre_terrain']:
            for item in pre_terrain:
                terrain += list(item)
        for item in step_data['terrain']:
            terrain += list(item)
        input = list(step_data['state']) + terrain + [step_data['env_action']]
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
    '''not used'''
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
        input = list(step_data['state']) + terrain + [step_data['env_action']]
        #print(len(input))
        inputs.append(input)

    #print(len(inputs))
    # print(inputs)
    # print(inputs[0])
    mydataset = MyDataset(inputs,labels)
    return mydataset

def create_pos_dataset(train_data_pos):
    '''*************************************************************************
    [function name]         create_pos_dataset
    [description]           创建positive数据集MyDataset类
    [input]                
    [output]                MyDataset | mydataset: torch.utils.data.Dataset
    [author]                brx, 2024
    [changelog]             （若修改过则必需注明）
    *************************************************************************'''
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
        # 去掉了pos
        input = list(step_data['state']) + terrain + [step_data['env_action']]
        #print(len(input))
        inputs.append(input)
        
    mydataset = MyDataset(inputs,labels)
    return mydataset

def create_neg_dataset(train_data_neg):
    '''*************************************************************************
    [function name]         create_neg_dataset
    [description]           创建negtive数据集MyDataset类
    [input]                
    [output]                MyDataset | mydataset: torch.utils.data.Dataset
    [author]                brx, 2024
    [changelog]             （若修改过则必需注明）
    *************************************************************************'''
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
        # 去掉了pos
        input = list(step_data['state']) + terrain + [step_data['env_action']]
        #print(len(input))
        inputs.append(input)
    """
    for k, data in enumerate(train_data):
        labels.append((0,0.0))
        inputs.append(data)
    """
    mydataset = MyDataset(inputs,labels)
    return mydataset

def create_pos_dataset_by_episode(train_data_pos):
    '''*************************************************************************
    [function name]         create_pos_dataset
    [description]           创建positive数据集MyDataset类
    [input]                
    [output]                MyDataset | mydataset: torch.utils.data.Dataset
    [author]                brx, 2024
    [changelog]             （若修改过则必需注明）
    *************************************************************************'''
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
    for k, episode_data in enumerate(train_data):
        # 获取最后一个step
        step_data=episode_data['episode'][-1]
        labels.append((step_data['label'],step_data['fall_dist']))
        terrain = []
        for pre_terrain in step_data['pre_terrain']:
            for item in pre_terrain:
                terrain += list(item)
        for item in step_data['terrain']:
            terrain += list(item)
        # 去掉了pos
        input = list(step_data['state']) + terrain + [step_data['env_action']]
        #print(len(input))
        inputs.append(input)
        
    mydataset = MyDataset(inputs,labels)
    return mydataset

def create_neg_dataset_by_episode(train_data_neg):
    '''*************************************************************************
    [function name]         create_neg_dataset
    [description]           创建negtive数据集MyDataset类
    [input]                
    [output]                MyDataset | mydataset: torch.utils.data.Dataset
    [author]                brx, 2024
    [changelog]             （若修改过则必需注明）
    *************************************************************************'''
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
    for k, episode_data in enumerate(train_data):
        # 获取最后一个step
        step_data=episode_data['episode'][-1]
        labels.append((step_data['label'],step_data['fall_dist']))
        terrain = []
        for pre_terrain in step_data['pre_terrain']:
            for item in pre_terrain:
                terrain += list(item)
        for item in step_data['terrain']:
            terrain += list(item)
        # 去掉了pos
        input = list(step_data['state']) + terrain + [step_data['env_action']]
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
    '''*************************************************************************
    [function name]         create_distill_dataset
    [description]           为相应数据增加标签 neg (0,0.0) pos (1,0.0)
    [input]                 dataset_path: 数据集路径 
                            dataset_type: 数据集类型 "total" "pos" "neg"
    [output]                MyDataset | mydataset: torch.utils.data.Dataset
    [author]                brx, 2024
    [changelog]             （若修改过则必需注明）
    *************************************************************************'''
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


# 下面这些似乎是没用了 >_<

"""
 需要一个函数，输入episode_path，中间输出step_data，(最终输出为dataset(label, input形式))(可能并非必要).
 
 对于episodes_data，每个元素包含键值对
 'episode'中的每个元素: 
    {
        'observation':observation, 
        'reward':reward, 
        'next_observation':next_observation,
        'truncated':truncated, # episode end
        'terminated':ternimated, # not finish
        'env_act':action, 
        'info':info
    },
info = {"position": (pos[0],pos[1]),
        'agent_i': self.agent_i,
        'lidar_pred': self.pred_i,
        'cur_step_poly': self.cur_step_poly,
        "weight": np.prod(self.weights),
        'cur_step_env_start': self.cur_start,
        'cur_step_env_end': self.cur_end,
        'next_env_actions': self.new_env_actions,
        'next_terrain_poly': self.new_terrain_poly,
        }
'terrain_poly',
'terrain_x',
'terrain_y',
'env_actions',
'all_pos',
'is_failure'.
 
 对于step_data，应包含
 label, 
 fall_dist, <--info['position'][0]
     max_i=min{info['current_step_env_end'], info['lidar_pred']} 
 pre_terrain, <--terrain_poly[max_i-11,max_i-2]
 terrain, <--terrain_poly[max_i-1]
 state, <--observation
 pos(判断是否为正样本), 
 env_action, <--env_act
 info <--info
"""
#for episode start
# this function does not fit the define of input 
def process_episode_data(episode_path, data_type=None):
    file_names = os.listdir(episode_path)
    raw_data_list = [
        episode_path +'/'+ file_name for file_name in file_names if file_name.endswith('.npy')]
    processed_episode_data=[]
    for raw_data in tqdm.tqdm(raw_data_list):
        episode_data=np.load(raw_data, allow_pickle=True)
        episode_data=episode_data[0] #(1,)->()
        window_size=11
        if len(episode_data['episode'])<window_size:
            # episode is too short
            continue
        else:
            for start_step in range(len(episode_data['episode'])%window_size, len(episode_data['episode']),window_size):
                steps_data=[]
                for step_index_in_window in range(0,window_size):
                    step_index_in_episode=step_index_in_window+start_step
                    step_data={}
                    is_failure=episode_data['episode'][step_index_in_episode]['truncated'] and episode_data['episode'][step_index_in_episode]['terminated']
                    step_data['label']=(is_failure, 0.0)
                    step_data['fall_dist']=episode_data['episode'][step_index_in_episode]['info']['position'][0] # need to understand the meaning of postion[0] if not fail
                    # print((episode_data['episode'][step_index_in_episode]['info']['cur_step_env_end'],episode_data['episode'][step_index_in_episode]['info']['lidar_pred']))
                    max_i=min(episode_data['episode'][step_index_in_episode]['info']['cur_step_env_end'], round(episode_data['episode'][step_index_in_episode]['info']['lidar_pred']))
                    step_data['pre_terrain']=[]
                    step_data['pre_terrain']+=(episode_data['terrain_poly'][i][0] for i in range(max_i-10,max_i-1))
                    # print(step_data['pre_terrain'])
                    step_data['terrain']=episode_data['terrain_poly'][max_i-1][0]
                    # print(step_data['terrain'])
                    step_data['state']=episode_data['episode'][step_index_in_episode]['observation']
                    step_data['pos']=[is_failure]
                    step_data['env_action']=episode_data['episode'][step_index_in_episode]['env_act']
                    step_data['info']=episode_data['episode'][step_index_in_episode]['info']
                    steps_data.append(step_data)
                processed_episode_data.append(steps_data)
    np.save('/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data/processed_data.npy', processed_episode_data, allow_pickle=True)


def create_my_episode_dataset(data_path, data_type):
    inputs=[]
    labels=[]
    data=[]
    if data_type=='total':
        data_pos=np.load(data_path[0], allow_pickle=True)
        data_neg=np.load(data_path[1], allow_pickle=True)
        data=list(data_pos)+list(data_neg)
    elif data_type=='pos':
        data=np.load(data_path, allow_pickle=True)
    elif data_type=='neg':
        data=np.load(data_path, allow_pickle=True)
    else:
        BG_RED = "\033[41m"
        RESET = "\033[0m"
        logging.error("UNEXPECTED DATA_TYPE!")  # 更通用的错误输出方式
    
    for episode_data in tqdm.tqdm(data):
        labels.append((episode_data[-1]['label'][0], episode_data[-1]['fall_dist']))
        episode_input=[]
        for step_data in episode_data:
            terrain=[]
            for pre_terrain in step_data['pre_terrain']:
                for item in pre_terrain:
                    terrain += list(item)
            for item in step_data['terrain']:
                terrain += list(item)
            step_input=list(step_data['state']) + terrain + [step_data['env_action']]
            episode_input.append(step_input)
        inputs.append(episode_input)
    print(len(inputs))
    print(len(inputs[0]))
    print(len(inputs[0][0]))
    mydataset = MyDataset(inputs,labels)
    return mydataset
#for episode end

#for step start
def process_step_data(episode_path, data_type=None):
    file_names = os.listdir(episode_path)
    raw_data_list = [
        episode_path +'/'+ file_name for file_name in file_names if file_name.endswith('.npy')]
    steps_data=[]
    print(len(raw_data_list))
    for raw_data in tqdm.tqdm(raw_data_list):
        episode_data=np.load(raw_data, allow_pickle=True)
        episode_data=episode_data[0] #(1,)->()
        for step_index_in_episode in range(0, len(episode_data['episode'])):
            step_data={}
            is_failure=episode_data['episode'][step_index_in_episode]['truncated'] and episode_data['episode'][step_index_in_episode]['terminated']
            step_data['label']=(is_failure, 0.0)
            step_data['fall_dist']=episode_data['episode'][step_index_in_episode]['info']['position'][0] # need to understand the meaning of postion[0] if not fail
            # print((episode_data['episode'][step_index_in_episode]['info']['cur_step_env_end'],episode_data['episode'][step_index_in_episode]['info']['lidar_pred']))
            max_i=min(episode_data['episode'][step_index_in_episode]['info']['cur_step_env_end'], round(episode_data['episode'][step_index_in_episode]['info']['lidar_pred']))
            step_data['pre_terrain']=[]
            step_data['pre_terrain']+=(episode_data['terrain_poly'][i][0] for i in range(max_i-11,max_i-2))
            # print(step_data['pre_terrain'])
            # print(episode_data['episode'][step_index_in_episode]['info']['cur_step_env_end'], round(episode_data['episode'][step_index_in_episode]['info']['lidar_pred']), max_i, len(episode_data['terrain_poly']))
            step_data['terrain']=episode_data['terrain_poly'][max_i-2][0]
            # print(step_data['terrain'])
            step_data['state']=episode_data['episode'][step_index_in_episode]['observation']
            step_data['pos']=[is_failure]
            step_data['env_action']=episode_data['episode'][step_index_in_episode]['env_act']
            step_data['info']=episode_data['episode'][step_index_in_episode]['info']
            steps_data.append(step_data)
            # print(len(steps_data))
    if data_type=='neg':
        np.save('/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data/processed_step_data_neg.npy', steps_data, allow_pickle=True)
    elif data_type=='pos':
        np.save('/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data/processed_step_data_pos.npy', steps_data, allow_pickle=True)
    else:
        logging.error("UNEXPECTED DATA_TYPE!")

            
def create_my_step_dataset(data_path, data_type):
    inputs=[]
    labels=[]
    data=[]
    if data_type=='total':
        data_pos=np.load(data_path[0], allow_pickle=True)
        data_neg=np.load(data_path[1], allow_pickle=True)
        data=list(data_pos)+list(data_neg)
    elif data_type=='pos':
        data=np.load(data_path, allow_pickle=True)
    elif data_type=='neg':
        data=np.load(data_path, allow_pickle=True)
    elif data_type == 'val':
        data_pos=np.load(data_path[0], allow_pickle=True)
        data_neg=np.load(data_path[1], allow_pickle=True)
        samples_data_pos = random.sample(data_pos,100)
        samples_data_neg = random.sample(data_neg,100)
        data=list(samples_data_pos)+list(samples_data_neg)
    else:
        BG_RED = "\033[41m"
        RESET = "\033[0m"
        logging.error("UNEXPECTED DATA_TYPE!")  
    # print(data)
    for step_data in tqdm.tqdm(data):
        labels.append((step_data['label'][0], step_data['fall_dist']))
        terrain=[]
        for pre_terrain in step_data['pre_terrain']:
            for item in pre_terrain:
                terrain += list(item)
        for item in step_data['terrain']:
            terrain += list(item)
        step_input=list(step_data['state']) + terrain + [step_data['env_action']]
        inputs.append(step_input)
    print(len(inputs))
    print(len(inputs[0]))
    mydataset = MyDataset(inputs,labels)
    return mydataset
        
# for step end 
        

def create_dataset_test(test_data_pos, test_data_neg):
    '''*************************************************************************
    [function name]         create_dataset_new
    [description]           
    [input]                
    [output]                MyDataset | mydataset: torch.utils.data.Dataset
    [author]                pjy, 2025.10.6
    [changelog]             （若修改过则必需注明）
    *************************************************************************'''
    np.random.seed(0)
    samples_data_pos = random.sample(test_data_pos,117)
    samples_data_neg = random.sample(test_data_neg,117)
    print(f'num of pos is {len(samples_data_pos)},num of neg is {len(samples_data_neg)}')
    train_data = samples_data_pos + samples_data_neg
    inputs = []
    labels = []

    for step_data in tqdm.tqdm(train_data):
        labels.append((step_data['label'],step_data['fall_dist']))
        terrain = []
        for pre_terrain in step_data['pre_terrain']:
            for item in pre_terrain:
                terrain += list(item)
        for item in step_data['terrain']:
            terrain += list(item)
        input = list(step_data['state']) + terrain + [step_data['env_action']]
        inputs.append(input)
    
    mydataset = MyDataset(inputs,labels)
    return mydataset
