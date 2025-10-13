import numpy as np
import os
import random
import math
def process_step(data):
    data = list(data)
    inputs = []
    for step_data in data:
        wind_idx = step_data['wind_idx'] - step_data['env_action']
        wind_mag = math.tanh( math.sin(0.02 * wind_idx) + math.sin(math.pi * 0.01 * wind_idx) ) * 10
    
        torque_idx = step_data['torque_idx'] - step_data['env_action']
        torque_mag = math.tanh( math.sin(0.02 * torque_idx)  + math.sin(math.pi * 0.01 * torque_idx)) 
    

        cur_input = list(step_data['state'])+ [wind_mag, torque_mag, step_data['env_action']]
        inputs += cur_input
    return inputs

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
            episode_obs = episode_data['obs']
            episode_action = episode_data['add_idx']
            episode_wind_idx = episode_data['wind_idx']
            episode_torque_idx = episode_data['add_idx']
            label = episode_data['crash']
            for i in range(len(episode_obs)):
                new_step_data = {}
                new_step_data['state'] = episode_obs[i]
                new_step_data['env_action'] = episode_action[i]
                new_step_data['label'] = label
                new_step_data['wind_idx'] = episode_wind_idx[i]
                new_step_data['torque_idx'] = episode_torque_idx[i]
            
                new_episode_data.append(new_step_data)
            print('step_num:',len(new_episode_data))
            for pre_steps in range(1,11):
                cur_pre_steps = new_episode_data[len(new_episode_data)-10-pre_steps:len(new_episode_data)-pre_steps]
                #print(len(cur_pre_steps))
                pre_inputs = process_step(cur_pre_steps)
                new_episode_data[len(new_episode_data)-pre_steps]['pre_inputs'] = pre_inputs
            positive_samples += new_episode_data[len(new_episode_data)-10:]
            #positive_samples += new_episode_data
        print(f"finish process processed_data {k}/{len(raw_data_list)}")
   
    num_train = int(p*len(positive_samples))
    num_test = int(0.5*(1-p)*len(positive_samples))
    print(num_train,num_test)
    
    random.shuffle(positive_samples)
    train_samples_pos = positive_samples
    val_samples_pos = positive_samples[num_train:num_train+num_test]
    test_samples_pos = positive_samples[num_train+num_test:]
    
    np.save(f'/mnt1/brx/dataset/train_dataset_pos_450_all.npy', train_samples_pos, allow_pickle=True)
    np.save(f'/mnt1/brx/dataset/val_dataset_pos_450_all.npy', val_samples_pos, allow_pickle=True)
    np.save(f'/mnt1/brx/dataset/test_dataset_pos_450_all.npy', test_samples_pos, allow_pickle=True)
    
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
    for k in range(500):
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
            episode_wind_idx = episode_data['wind_idx']
            episode_torque_idx = episode_data['add_idx']
            label = episode_data['crash']
            for i in range(len(episode_obs)):
                new_step_data = {}
                new_step_data['state'] = episode_obs[i]
                new_step_data['env_action'] = episode_action[i]
                new_step_data['label'] = label
                new_step_data['wind_idx'] = episode_wind_idx[i]
                new_step_data['torque_idx'] = episode_torque_idx[i]
            
                new_episode_data.append(new_step_data)
            #positive_samples += new_episode_data[len(new_episode_data)-10:]
            
            for pre_steps in range(1,len(new_episode_data)-10):
                cur_pre_steps = new_episode_data[len(new_episode_data)-10-pre_steps:len(new_episode_data)-pre_steps]
                pre_inputs = process_step(cur_pre_steps)
                new_episode_data[len(new_episode_data)-pre_steps]['pre_inputs'] = pre_inputs
            positive_samples += new_episode_data
            
        print(f"finish process processed_data {k}/{len(raw_data_list)}")
   
    num_train = int(p*len(positive_samples))
    num_test = int(0.5*(1-p)*len(positive_samples))
    print(num_train,num_test)
    
    random.shuffle(positive_samples)
    #train_samples_pos = positive_samples
    train_samples_pos = positive_samples[:num_train]
    val_samples_pos = positive_samples[num_train:num_train+num_test]
    test_samples_pos = positive_samples[num_train+num_test:]
    
    np.save(f'/mnt1/brx/dataset/train_dataset_neg_450_all.npy', train_samples_pos, allow_pickle=True)
    np.save(f'/mnt1/brx/dataset/val_dataset_neg_450_all.npy', val_samples_pos, allow_pickle=True)
    np.save(f'/mnt1/brx/dataset/test_dataset_neg_450_all.npy', test_samples_pos, allow_pickle=True)
    

#log_dir_pos = '/mnt1/brx/Rocketdata/positive/'
log_dir_pos = '/mnt1/brx/Rocketdata/new_dataset_450/positive/'
#build_pos_dataset(log_dir_pos,p=0.95)
log_dir_neg = '/mnt1/brx/Rocketdata/new_dataset_450/negative/'
build_neg_dataset(log_dir_neg,p=0.95)
