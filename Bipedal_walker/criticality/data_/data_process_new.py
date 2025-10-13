'''*************************************************************************
【文件名】                 （必需）
【功能模块和目的】         处理由collect_data_new.py生成的数据，将数据处理成适合训练的格式
                        input: raw_data/pos/transitions_{epoch}.npy'
                        output: processed_data/pos/transitions_{epoch}.npy
【开发者及日期】           （必需）
【更改记录】               （若修改过则必需注明）
*************************************************************************'''
import tqdm
import numpy as np
import os
import shutil
# raw_data = np.load('data/raw_data/transitions_0.npy', allow_pickle=True)

"""
print(type(raw_data))
print(len(raw_data))
print(raw_data[0])
print(raw_data[0]['episode'])
print(raw_data[0].keys())
for item in raw_data[0]['env_actions']:
    print(item)
"""
max_steps = 160
TERRAIN_STEP = 14 / 30.0

def data_process_(data_path=None,k=None, processed_data_pos_path=None):
    raw_data = np.load(data_path, allow_pickle=True)
    print(f'load from {data_path},length is {len(raw_data)}')
    processed_data = []
    positive_nums = 0
    # episodes
    print('processing data...')
    for item in tqdm.tqdm(raw_data):
        # print(f'Process episode {l}/{len(raw_data)}')
        episode = item['episode']
        env_actions = item['env_actions']
        is_failure = item['is_failure']
        # fall_pos = item['fall_pos'] # 没有用到
        # steps
        # print("steps: ", len(episode))
        new_episode_data = []
        for i,step_data in enumerate(episode):
            # print(i)
            # print(f'Process step {i} of episode {k}')
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

            if pred_i >= len(env_actions):
                continue
            pre_terrain_num = np.linspace(cur_i,pred_i,10,endpoint=True)
            pre_terrain_num = [int(item) for item in pre_terrain_num]
            pre_terrain = []
            for n in range(len(pre_terrain_num)-1):
                pre_terrain_info = env_actions[pre_terrain_num[n]]
                next_terrain_info = env_actions[pre_terrain_num[n+1]]
                if pre_terrain_info['terrain_type'] == 1 or 'env_poly' not in pre_terrain_info:
                    pre_terrain.append([(pre_terrain_info['start_x'],pre_terrain_info['current_y']),
                                        (next_terrain_info['start_x'],next_terrain_info['current_y']),
                                        (next_terrain_info['start_x'],next_terrain_info['current_y']),
                                        (pre_terrain_info['start_x'],pre_terrain_info['current_y'])])
                else:
                    pre_terrain.append(pre_terrain_info['env_poly'])


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
            new_step_data['pre_terrain'] = pre_terrain
            new_step_data['env_action'] = cur_action
            new_step_data['agent_i'] = cur_i
            new_step_data['pred_i'] = pred_i
            new_step_data['info'] = other_info
            new_step_data['is_failure'] = is_failure
            if is_failure and (i==len(episode)-1):
                new_step_data['label'] = 1
            else:
                new_step_data['label'] = 0
            new_step_data['fall_dist'] = cur_step_poly[1][0]-cur_pos[0]
            new_episode_data.append(new_step_data)
            # print(new_step_data)
        if is_failure:
            positive_nums += 1
        else:
            pass

        """
        if is_failure:
            positive_nums += 1
            j = len(processed_data)-1
            while j >= 0:
                processed_data[j]['fall_dist'] = fall_pos - processed_data[j]['pos'][0]
                j -= 1
        """
                
        processed_data += new_episode_data
        # print(new_episode_data)
    #np.save(f'/root/autodl-tmp/data/processed_data/transitions_{k}.npy', processed_data, allow_pickle=True)
    # np.save(f'/home/yjx/tta_new/data/processed_data_pos/transitions_{k}.npy', processed_data, allow_pickle=True)
    if not os.path.exists(processed_data_pos_path):
        os.makedirs(processed_data_pos_path)
    np.save(processed_data_pos_path+f'/transitions_{k}.npy', processed_data, allow_pickle=True)
    print(f'length of processed data is {len(processed_data)},length of positive samples is {positive_nums}')
    positive_nums=0
    # print(processed_data)
    # return processed_data


def data_process(log_dir, lable='pos'):
    # log_dir = 'data/raw_data/'
    file_names = os.listdir(log_dir)
    raw_data_list = [
        log_dir + file_name for file_name in file_names if file_name.endswith('.npy')]
    processed_data_path = '/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data'+'_'+lable+'/'
    for k,raw_data_path in enumerate(raw_data_list):
        data_process_(raw_data_path,k,processed_data_path)
        print(f"finish process raw_data {k+1}/{len(raw_data_list)}")

    # 汇总所有的triansitions_{k}.npy文件
    processed_data_list = []
    for k in range(len(raw_data_list)):
        processed_data_path_k = processed_data_path + f'transitions_{k}.npy'
        if os.path.exists(processed_data_path_k):
            processed_data = np.load(processed_data_path_k, allow_pickle=True)
            processed_data_list += processed_data.tolist()
    np.save(processed_data_path+'transitions_all.npy', processed_data_list, allow_pickle=True)

def clear_processed_data_pos(processed_data_pos_path_root=None):
    processed_data_pos_path_root='/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data_pos/'
    file_names = os.listdir(processed_data_pos_path_root)
    processed_data_pos_list=[
        processed_data_pos_path_root + file_name for file_name in file_names if file_name.endswith('.npy') and not file_name.endswith('all.npy')]
    processed_data_pos_new_path_root='/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data_pos_new/'
    processed_data_neg_new_path_root='/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data_neg_new/'
    for k, processed_data_pos_path in enumerate(processed_data_pos_list):
        processed_data_pos=np.load(processed_data_pos_path, allow_pickle=True)
        processed_data_pos_new=[]
        processed_data_neg_new=[]
        for i,step_data in enumerate(processed_data_pos):
            if step_data['label']==0:
                processed_data_neg_new.append(step_data)
            else:
                processed_data_pos_new.append(step_data)
        
        np.save(processed_data_pos_new_path_root+f'transitions_{k}.npy', processed_data_pos_new, allow_pickle=True)
        np.save(processed_data_neg_new_path_root+f'transitions_{k+2064}.npy', processed_data_neg_new, allow_pickle=True)      

    # 汇总所有的triansitions_{k}.npy文件for pos
    processed_data_pos_new_list = []
    for k in range(len(processed_data_pos_list)):
        processed_data_new_path_k = processed_data_pos_new_path_root + f'transitions_{k}.npy'
        if os.path.exists(processed_data_new_path_k):
            processed_data_pos = np.load(processed_data_new_path_k, allow_pickle=True)
            processed_data_pos_new_list += processed_data_pos.tolist()
    np.save(processed_data_pos_new_path_root+'transitions_all.npy', processed_data_pos_new_list, allow_pickle=True)

    # 汇总所有的triansitions_{k}.npy文件for neg
    processed_data_neg_new_list = []
    for k in tqdm.tqdm(range(len(processed_data_pos_list)+2064)):
        processed_data_new_path_k = processed_data_neg_new_path_root + f'transitions_{k}.npy'
        if os.path.exists(processed_data_new_path_k):
            processed_data_neg = np.load(processed_data_new_path_k, allow_pickle=True)
            processed_data_neg_new_list += processed_data_neg.tolist()
    print('saving processed_data_neg_new_list')
    np.save(processed_data_neg_new_path_root+'transitions_all.npy', processed_data_neg_new_list, allow_pickle=True)
    print('processed_data_neg_new_list saved')



def copy_files_with_extension(src_dir=None, dst_dir=None, extension=None):
    """
    复制具有给定扩展名的文件从源目录到目标目录。
    参数:
    src_dir (str): 源文件夹路径。
    dst_dir (str): 目标文件夹路径。
    extension (str): 文件扩展名（例如 '.txt' 或者 '.jpg'）。
    """
    extension='.npy'
    src_dir='/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data_neg'
    dst_dir='/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data_neg_new'
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    # 遍历源文件夹中的所有文件
    for root, dirs, files in os.walk(src_dir):
        for file_name in tqdm.tqdm(files):
            if file_name.endswith(extension) and not file_name.endswith('all.npy'):
                src_file_path = os.path.join(root, file_name)
                dst_file_path = os.path.join(dst_dir, file_name)
                try:
                    shutil.copy2(src_file_path, dst_file_path)
                    # print(f"成功复制文件: {src_file_path} 到 {dst_file_path}")
                except Exception as e:
                    print(f"无法复制文件 {src_file_path}: {e}")

# lable = 'neg'
# log_dir = '/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/raw_data/'+lable+'/'
# data_process(log_dir, lable=lable)
# lable = 'pos'
# log_dir = '/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/raw_data/'+lable+'/'
# data_process(log_dir, lable=lable)

# copy_files_with_extension()

clear_processed_data_pos()