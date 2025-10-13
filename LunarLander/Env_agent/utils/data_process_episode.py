import numpy as np
import os
import random

def process_step(data_):
    data_ = list(data_)
    a = []
    for step_ in data_:
        a += step_['input']
        print('step input',len(step_['input']))
    print('a',len(a))
    return a


def build_pos_dataset(log_dir,p):
    # log_dir = 'tta/data/processed_data/'
    # p: proportion of train dataset 0.7
    file_names = os.listdir(log_dir)
    raw_data_list = [
        log_dir + file_name for file_name in file_names if file_name.endswith('.npy')]
    positive_samples = []
    negative_samples = []
    neg_pos_ratio = 0
    train_neg_num = 0
    for k in range(len(raw_data_list)):
        raw_data_path = raw_data_list[k]
        processed_data = np.load(raw_data_path, allow_pickle=True)
        """
        if k>34:
            break
        """
        print(len(positive_samples))
        
        for episode_data in processed_data:
            """
             data_episode={'obs':[],'action':[],'wind_idx':[],'torque_idx':[],'add_idx':[],'reward':[],'crash':0}
        obs,_= env.reset()
            """
            new_episode_data = []
            episode_obs = episode_data['obs']  # at-1
            episode_action = episode_data['add_idx']         # at
            #episode_action = [0] + episode_action[:-1]      # at-1
            episode_wind_idx = episode_data['wind_idx']      # at
            #episode_wind_idx = episode_data['wind_idx'] - episode['wind_idx']  # at-1
            episode_torque_idx = episode_data['torque_idx']  # 
            label = episode_data['crash']
            print(episode_action)
            for i in range(len(episode_obs)-1):
                new_step_data = {}
                new_step_data['state'] = episode_obs[i]
                new_step_data['env_action'] = episode_action[i]
                new_step_data['label'] = label
                new_step_data['wind_idx'] = episode_wind_idx[i]
                new_step_data['torque_idx'] = episode_torque_idx[i]
                #print('wind_idx',new_step_data['wind_idx'],'torque_idx',new_step_data['torque_idx'])
            
                new_episode_data.append(new_step_data)
            #positive_samples.append(new_episode_data[len(new_episode_data)-11:])
            step_num = len(new_episode_data) // 11
            if step_num > 5:
                step_num = 5
            for j in range(step_num):
                positive_samples.append(new_episode_data[len(new_episode_data)-11*(j+1):len(new_episode_data)-11*j])
            
        print(f"finish process processed_data {k}/{len(raw_data_list)}")
   
    num_train = int(p*len(positive_samples))
    num_test = int((1-p)*len(positive_samples))
    print(num_train,num_test)
    
    random.shuffle(positive_samples)
    train_samples_pos = positive_samples[:num_train]
    #val_samples_pos = positive_samples[num_train:num_train+num_test]
    test_samples_pos = positive_samples[num_train:]
    
    #np.save(f'/mnt1/brx/dataset/train_dataset_pos_episode_final5.npy', train_samples_pos, allow_pickle=True)
    #np.save(f'/mnt1/brx/dataset/val_dataset_pos_episode_450_new.npy', val_samples_pos, allow_pickle=True)
    #np.save(f'/mnt1/brx/dataset/test_dataset_pos_episode_final5.npy', test_samples_pos, allow_pickle=True)
    
def build_neg_dataset(log_dir,p):
    # log_dir = 'tta/data/processed_data/'
    # p: proportion of train dataset 0.7
    file_names = os.listdir(log_dir)
    raw_data_list = [
        log_dir + file_name for file_name in file_names if file_name.endswith('.npy')]
    positive_samples = []
    negative_samples = []
    neg_pos_ratio = 0
    train_neg_num = 0
    print(len(raw_data_list))
    for k in range(1000):
        raw_data_path = raw_data_list[k]
        processed_data = np.load(raw_data_path, allow_pickle=True)
        """
        if k>34:
            break
        """
        print(len(positive_samples))
        
        for episode_data in processed_data:
            """
             data_episode={'obs':[],'action':[],'wind_idx':[],'torque_idx':[],'add_idx':[],'reward':[],'crash':0}
        obs,_= env.reset()
            """
            new_episode_data = []
            episode_obs = episode_data['obs']
            episode_action = episode_data['add_idx']
            #episode_action = [0] + episode_action[:-1]
            episode_wind_idx = episode_data['wind_idx']
            episode_torque_idx = episode_data['torque_idx']
            label = episode_data['crash']
            for i in range(len(episode_obs)):
                new_step_data = {}
                new_step_data['state'] = episode_obs[i]
                new_step_data['env_action'] = episode_action[i]
                new_step_data['label'] = label
                new_step_data['wind_idx'] = episode_wind_idx[i]
                new_step_data['torque_idx'] = episode_torque_idx[i]
            
                new_episode_data.append(new_step_data)
            step_num = len(new_episode_data) // 11
            for j in range(step_num):
                positive_samples.append(new_episode_data[len(new_episode_data)-11*(j+1):len(new_episode_data)-11*j])
            
        print(f"finish process processed_data {k}/{len(raw_data_list)}")
   
    num_train = int(p*len(positive_samples))
    num_test = int(0.5*(1-p)*len(positive_samples))
    print(num_train,num_test)
    
    random.shuffle(positive_samples)
    #train_samples_pos = positive_samples
    train_samples_pos = positive_samples[:num_train]
    val_samples_pos = positive_samples[num_train:num_train+num_test]
    test_samples_pos = positive_samples[num_train+num_test:]
    
    np.save(f'/mnt1/brx/dataset/train_dataset_neg_episode_450_new.npy', positive_samples, allow_pickle=True)
    np.save(f'/mnt1/brx/dataset/val_dataset_neg_episode_450_new.npy', val_samples_pos, allow_pickle=True)
    np.save(f'/mnt1/brx/dataset/test_dataset_neg_episode_450_new.npy', test_samples_pos, allow_pickle=True)
    

log_dir_pos = '/mnt1/brx/Rocketdata/new_dataset_450/positive/'
build_pos_dataset(log_dir_pos,p=0.9)
log_dir_neg = '/mnt1/brx/Rocketdata/new_dataset_450/negative/'
#build_neg_dataset(log_dir_neg,p=0.9)
