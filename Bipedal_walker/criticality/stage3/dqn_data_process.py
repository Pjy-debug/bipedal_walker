import numpy as np
import os
import random
def build_dataset(log_dir,p):
    # log_dir = 'tta/data/processed_data/'
    # p: proportion of train dataset 0.7
    file_names = os.listdir(log_dir)
    raw_data_list = [
        log_dir + file_name for file_name in file_names if file_name.endswith('.npy')]
    replay_buffer = []
    for i in range(len(raw_data_list)):
        raw_data_path = raw_data_list[i]
        processed_data = np.load(raw_data_path, allow_pickle=True)
       
        for trajectory in processed_data:
            total_inputs = trajectory['inputs'] 
            total_all_inputs = trajectory['all_inputs'] 
            total_rewards = trajectory['rewards'] 
            total_dones = trajectory['dones'] 
            total_actions = trajectory['actions'] 
            total_terrain_states = trajectory['terrain_states'] 
            
            for k in range(len(total_inputs)):
                item = {}
                if k == len(total_inputs) - 1:
                    next_inputs = total_inputs[-1]
                    next_all_inputs = total_all_inputs[-1]
                else:
                    next_inputs = total_inputs[k+1]
                    next_all_inputs = total_all_inputs[k+1]
                item['input'] = total_inputs[k]
                item['action'] = total_actions[k]
                #item['reward'] = total_rewards[k]
                item['reward'] = total_rewards[k]
                #print(item['reward'])
                item['next_input'] = next_inputs
                item['done'] =  total_dones[k]
                item['terrain_state'] = total_terrain_states[k]
                item['all_inputs'] =  total_all_inputs[k]
                item['next_al_inputs'] = next_all_inputs
                replay_buffer.append(item)

        
        print(f"finish process processed_data {i}/{len(raw_data_list)}")
    
    print(len(replay_buffer))
    

    #np.save(f'/home/yjx/tta_new/data/dataset/dqn_data.npy', replay_buffer, allow_pickle=True)
    np.save(f'/home/yjx/tta_new/data/dataset/dqn_data_neg.npy', replay_buffer, allow_pickle=True)

#log_dir = '/home/yjx/tta_new/data/raw_data_dqn/'
log_dir = '/mnt/yjx/dqn_neg_raw_data/'
build_dataset(log_dir,p=0.95)