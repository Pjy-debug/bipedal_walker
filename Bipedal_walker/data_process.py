import numpy as np
import os
raw_data = np.load('data/raw_data/transitions_0.npy', allow_pickle=True)

"""
print(type(raw_data))
print(len(raw_data))
print(raw_data[0])
print(raw_data[0]['episode'])
print(raw_data[0].keys())
for item in raw_data[0]['env_actions']:
    print(item)
"""
max_steps = 2000
TERRAIN_STEP = 14 / 30.0

def data_process_(data_path,k):
    raw_data = np.load(data_path, allow_pickle=True)
    processed_data = []
    # episodes
    for item in raw_data:
        episode = item['episode']
        env_actions = item['env_actions']
        is_failure = item['is_failure']
        fall_pos = item['fall_pos']
        # steps
        # print(len(episode))
        for i,step_data in enumerate(episode):
            # print(i)
            new_step_data = {}
            cur_pos = step_data['info']['position']
            cur_i = int(step_data['info']['agent_i'])
            # print(step_data['info'].keys())
            if 'lidar_pred' in step_data['info'].keys():
                pred_i = int(step_data['info']['lidar_pred'])
            else:
                continue
            cur_terrain_info = env_actions[cur_i]
            # print(cur_terrain_info.keys())
            if 'input_action' in cur_terrain_info.keys():
                cur_action = cur_terrain_info['input_action']
            else:
                continue
            cur_step_poly = step_data['info']['cur_step_poly']
            if 'action' in cur_terrain_info.keys():
                other_info = {'action': cur_terrain_info['action'], 'type': cur_terrain_info['terrain_type']}
            else:
                other_info = {'action': [], 'type': cur_terrain_info['terrain_type']}

            """
            if cur_terrain_info['terrain_type'] == 0:
                if cur_i + 1 < len(env_actions):
                    next_terrain_info = env_actions[cur_i+1]
                    cur_terrain = [(cur_terrain_info['start_x'],cur_terrain_info['current_y']),
                                   (next_terrain_info['start_x'],cur_terrain_info['current_y']),
                                   (next_terrain_info['start_x'],cur_terrain_info['current_y']),
                                   (next_terrain_info['start_x'],cur_terrain_info['current_y'])]
                    if 'action' in cur_terrain_info.keys():
                        other_info = {'action': cur_terrain_info['action'], 'type': 0}
                    else:
                        other_info = {'action': [], 'type': 0}
                else:
                    continue
            elif cur_terrain_info['terrain_type'] == 1:
                if 'env_poly' not in cur_terrain_info.keys():
                    continue
                cur_terrain = cur_terrain_info['env_poly']
                if 'action' in cur_terrain_info.keys():
                    other_info = {'action':cur_terrain_info['action'],'type':1}
                else:
                    other_info = {'action': [], 'type': 1}
            elif cur_terrain_info['terrain_type'] == 2:
                if 'env_poly' not in cur_terrain_info.keys():
                    continue
                cur_terrain = cur_terrain_info['env_poly']
                if 'action' in cur_terrain_info.keys():
                    other_info = {'action':cur_terrain_info['action'],'type':2}
                else:
                    other_info = {'action': [], 'type': 2}
            """

            new_step_data['state'] = step_data['observation']
            new_step_data['pos'] = list(cur_pos)
            new_step_data['terrain'] = cur_step_poly
            new_step_data['env_action'] = cur_action
            new_step_data['agent_i'] = cur_i
            new_step_data['pred_i'] = pred_i
            new_step_data['info'] = other_info
            new_step_data['label'] = is_failure
            new_step_data['fall_dist'] = -1
            processed_data.append(new_step_data)
            # print(new_step_data)
        
        if is_failure:
            j = len(processed_data)-1
            while j >= 0:
                processed_data[j]['fall_dist'] = fall_pos - processed_data[j]['pos'][0]
                j -= 1
        
        
        """
        #if len(episode) < max_steps:
        if is_failure:
            fall_pos = processed_data[-1]['pos'][0]
            j = len(processed_data)-1
            # print(processed_data[j]['terrain'][0][0],fall_pos)
            while processed_data[j]['terrain'][0][0] > fall_pos and j>=0:
                j -= 1
            # print(j)
            # print(processed_data[j]['terrain'][0][0], fall_pos)
            final = j
            processed_data = processed_data[:final+1]
            while j >= 0 and processed_data[j]['terrain'] == processed_data[final]['terrain']:
                processed_data[j]['label'] = 1
                processed_data[j]['fall_dist'] = processed_data[final]['terrain'][0][0] - processed_data[j]['pos'][0]
                # print(j,processed_data[j]['terrain'],processed_data[j]['fall_dist'],int(processed_data[j]['fall_dist']/TERRAIN_STEP))
                j -= 1
        """

    np.save(f'/home/yjx/tta_new/data/processed_data/transitions_{k}.npy', processed_data, allow_pickle=True)
    #np.save(f'/root/autodl-tmp/data/processed_data/transitions_{k}.npy', processed_data, allow_pickle=True)
    # print(processed_data)
    # return processed_data


def data_process(log_dir):
    # log_dir = 'data/raw_data/'
    file_names = os.listdir(log_dir)
    raw_data_list = [
        log_dir + file_name for file_name in file_names if file_name.endswith('.npy')]
    for k,raw_data_path in enumerate(raw_data_list):
        data_process_(raw_data_path,k)
        print(f"finish process raw_data {k+1}/{len(raw_data_list)}")

log_dir = '/home/yjx/tta_new/data/raw_data/'
data_process(log_dir)