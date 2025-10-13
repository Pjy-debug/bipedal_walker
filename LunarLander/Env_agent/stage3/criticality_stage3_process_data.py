import numpy as np
import os
import random
import math
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def process_step(data):
    data = list(data)
    a = []
    for step_data in data:
        assert len(step_data['input']) == 11
        a += step_data['input']

    return a

'''
def get_episode_data(new_episode_data):
    dqn_episode_data = []
    for pre_steps in range(11, len(new_episode_data)):
        # print(pre_steps)
        cur_pre_steps = new_episode_data[pre_steps - 10:pre_steps]
        #print(len(cur_pre_steps))
        pre_inputs = process_step(cur_pre_steps)
        #print('pre_inputs:',len(pre_inputs))
        cur_step_data = new_episode_data[pre_steps]
        cur_inputs = pre_inputs + cur_step_data['input']
        #print('cur_inputs:',len(cur_inputs))
        # print(len(cur_step_data['all_input']))
        cur_all_inputs = [pre_inputs + k for k in cur_step_data['all_input']]
        #print('cur_all_inputs:',len(cur_all_inputs))
        next_inputs = pre_inputs[11:] + cur_step_data['input'] + cur_step_data['next_input']
        #print('next_inputs:',len(next_inputs))
        # print(len(cur_step_data['next_all_input']))
        next_all_inputs = [pre_inputs[11:] + cur_step_data['input'] + m for m in cur_step_data['next_all_input']]
        # print('next_all_inputs:',len(next_all_inputs))
        cur_step_data['input'] = cur_inputs
        cur_step_data['all_input'] = cur_all_inputs
        cur_step_data['next_input'] = next_inputs
        cur_step_data['next_all_input'] = next_all_inputs
        if pre_steps == len(new_episode_data) - 1:
            cur_step_data['label'] = 1
        else:
            cur_step_data['label'] = 0
        dqn_episode_data.append(cur_step_data)
        return dqn_episode_data

'''
def build_pos_dataset(log_dir, p):
    # log_dir = 'tta/data/processed_data/'
    # p: proportion of train dataset 0.7
    file_names = os.listdir(log_dir)
    raw_data_list = [
        log_dir + file_name for file_name in file_names if file_name.endswith('.npy')]
    positive_samples = []
    negative_samples = []
    neg_pos_ratio = 0
    train_neg_num = 0
    replay_buffer = []
    print(len(raw_data_list))
    for k in range(617):
        raw_data_path = raw_data_list[k]
        processed_data = np.load(raw_data_path, allow_pickle=True)
        """
        if k>34:
            break
        """
        # print('len pos',len(positive_samples))

        for episode_data in processed_data:
            """
             data_episode={'obs':[],'action':[],'wind_idx':[],'torque_idx':[],'add_idx':[],'reward':[],'crash':0}
        obs,_= env.reset()
            """
            # print(len(processed_data))
            episode_obs = episode_data['obs']  #
            episode_action = episode_data['add_idx']  # at
            episode_wind_idx = episode_data['wind_idx']  # at后
            episode_torque_idx = episode_data['torque_idx']
            label = episode_data['crash']
            episode_input = []
            episode_all_input = []
            new_episode_data = []
            # print(len(episode_obs))
            for i in range(len(episode_obs)):
                """
                wind_idx = episode_wind_idx[i]   # at
                torque_idx = episode_torque_idx[i]
                add_idx = episode_action[i]
                cur_all_input = []
                for j in range(17):
                    wind_idx_j = wind_idx - add_idx + j
                    torque_idx_j = torque_idx - add_idx + j
                    wind_mag_j = math.tanh( math.sin(0.02 * wind_idx_j) + math.sin(math.pi * 0.01 * wind_idx_j)) * 10
                    torque_mag_j = math.tanh( math.sin(0.02 * torque_idx_j)  + math.sin(math.pi * 0.01 * torque_idx_j)) 

                    # st,在st条件下执行的at
                    cur_all_input.append(list(episode_obs[i]) + [wind_mag_j, torque_mag_j, j])
                cur_input = cur_all_input[add_idx]
                episode_input.append(cur_input)
                episode_all_input.append(cur_all_input)
                """

                wind_idx = episode_wind_idx[i] - episode_action[i]  # at-1
                wind_mag = math.tanh(math.sin(0.02 * wind_idx) + math.sin(math.pi * 0.01 * wind_idx)) * 10

                torque_idx = episode_torque_idx[i] - episode_action[i]
                torque_mag = math.tanh(math.sin(0.02 * torque_idx) + math.sin(math.pi * 0.01 * torque_idx))

                cur_input = list(episode_obs[i]) + [wind_mag, torque_mag, episode_action[i]]
                cur_all_input = [list(episode_obs[i]) + [wind_mag, torque_mag] + [j] for j in range(17)]
                episode_input.append(cur_input)
                episode_all_input.append(cur_all_input)

            for i in range(len(episode_obs)):
                new_step_data = {}
                if i == len(episode_obs) - 1:
                    new_step_data['next_input'] = episode_input[i]
                    new_step_data['next_all_input'] = episode_all_input[i]
                    new_step_data['reward'] = 0
                    new_step_data['done'] = 1
                else:
                    new_step_data['next_input'] = episode_input[i + 1]
                    new_step_data['next_all_input'] = episode_all_input[i + 1]
                    new_step_data['reward'] = 0
                    new_step_data['done'] = 0

                new_step_data['input'] = episode_input[i]
                new_step_data['all_input'] = episode_all_input[i]
                new_step_data['action'] = episode_action[i]

                new_episode_data.append(new_step_data)

            dqn_episode_data = get_episode_data(new_episode_data)
            # print(len(new_episode_data))
            # replay_buffer += new_episode_data[len(new_episode_data)-10:]

            # print(len(dqn_episode_data))
            # positive_samples += dqn_episode_data

            replay_buffer += dqn_episode_data

        print(len(replay_buffer))

    np.save(f'/mnt/mnt1/tyy/data/dqn_episode_dataset_pos_reward0.npy', replay_buffer, allow_pickle=True)



def get_episode_data(new_episode_data):
    """
    根据新的 episode 数据格式生成 DQN 格式的数据集。
    """
    dqn_episode_data = []
    if len(new_episode_data) < 11:
        return []
    print(len(new_episode_data))

    for pre_steps in range(10, len(new_episode_data)):
        cur_pre_steps = new_episode_data[pre_steps - 10:pre_steps]
        pre_inputs = process_step(cur_pre_steps)
        cur_step_data = new_episode_data[pre_steps]
        
        # 核心修复: 创建一个新的字典，而不是修改原有的字典
        new_info = {}
        
        # 当前状态输入：前10个步骤的输入 + 当前步骤的输入
        new_info['input'] = pre_inputs + cur_step_data['input']
        
        # 修复逻辑：处理 all_input 的拼接
        cur_all_inputs = [pre_inputs + k if isinstance(k, list) else pre_inputs + k.tolist() for k in cur_step_data['all_input']]
        new_info['all_input'] = cur_all_inputs
        
        # 下一个状态的输入：当前滑动窗口的输入 + 下一个步骤的输入
        if pre_steps < len(new_episode_data) - 1:
            next_step_data = new_episode_data[pre_steps + 1]
            next_inputs = pre_inputs[11:] + cur_step_data['input'] + next_step_data['input']
            next_all_inputs = [pre_inputs[11:] + cur_step_data['input'] + (m if isinstance(m, list) else m.tolist()) for m in next_step_data['all_input']]
        else:
            # 对于最后一个步骤，下一个状态与当前状态相同
            next_inputs = new_info['input']
            next_all_inputs = new_info['all_input']

        new_info['action'] = cur_step_data['action']
        new_info['reward'] = cur_step_data['reward']
        new_info['done'] = cur_step_data['done']
        new_info['next_input'] = next_inputs
        new_info['next_all_input'] = next_all_inputs

        if pre_steps == len(new_episode_data) - 1:
            new_info['label'] = 1
        else:
            new_info['label'] = 0
            
        dqn_episode_data.append(new_info)
        
    return dqn_episode_data
def build_pos_dataset(log_dir):
    """
    根据新的数据格式处理数据集。
    Args:
        log_dir (str): 包含 .npy 文件的目录路径。
    """
    if not os.path.isdir(log_dir):
        print(f"错误：目录不存在或不是目录 {log_dir}")
        return

    file_names = os.listdir(log_dir)
    raw_data_list = [
        os.path.join(log_dir, file_name) for file_name in file_names if file_name.endswith('.npy')]
    
    replay_buffer = []
    
    print(f"找到 {len(raw_data_list)} 个数据文件。")

    for raw_data_path in raw_data_list:
        try:
            processed_data = np.load(raw_data_path, allow_pickle=True)
        except Exception as e:
            print(f"警告：无法加载文件 {os.path.basename(raw_data_path)}。错误：{e}")
            continue

        print(f"正在处理文件: {os.path.basename(raw_data_path)}, 包含 {len(processed_data)} 个 episode。")

        for episode_data in processed_data:
            episode_action = episode_data['total_actions']
            episode_all_inputs = episode_data['total_all_inputs']
            episode_rewards = episode_data['total_rewards']
            episode_dones = episode_data['total_dones']
            
            # 增加数据完整性检查
            if not (len(episode_action) == len(episode_all_inputs) == len(episode_rewards) == len(episode_dones)):
                print(f"警告：跳过一个数据不完整的episode，因为其关键列表长度不一致。")
                continue

            new_episode_data = []
            num_steps = len(episode_all_inputs)

            for i in range(num_steps):
                new_step_data = {}
                
                # 从 total_all_inputs 中提取当前步骤的输入数据
                # episode_all_inputs[i] 包含了 17 个可能动作的所有输入
                # episode_action[i] 是实际执行的动作索引
                # 从 total_all_inputs 中提取当前步骤的输入数据
                try:
                    cur_input_raw = episode_all_inputs[i][episode_action[i]]
                    # 检查当前输入的长度是否为 11

                    cur_input = list(cur_input_raw[-11:])  # 只取最后 11 个元素
                except (IndexError, TypeError) as e:
                    print(f"警告：处理第 {i} 步时发生错误 ({e})，跳过该步。")
                    continue
                
                new_step_data['input'] = cur_input[-11:]  # 只取最后 11 个元素
                new_step_data['all_input'] = list(episode_all_inputs[i][-11:])
                new_step_data['action'] = episode_action[i]
                new_step_data['reward'] = episode_rewards[i]
                new_step_data['done'] = episode_dones[i]

                if i < num_steps - 1:
                    new_step_data['next_input'] = list(episode_all_inputs[i + 1][episode_action[i + 1]][-11:])
                    new_step_data['next_all_input'] = list(episode_all_inputs[i + 1][-11:])
                else:
                    new_step_data['next_input'] = cur_input[-11:]
                    new_step_data['next_all_input'] = list(episode_all_inputs[i][-11:])

                new_episode_data.append(new_step_data)
            
            processed_episode = get_episode_data(new_episode_data)
            replay_buffer.extend(processed_episode)
            print(f"当前回放缓冲区大小: {len(replay_buffer)}")
            if (len(replay_buffer)) > 300000:
                break
        if (len(replay_buffer)) > 300000:
            break

    output_file_path = f'/mnt/mnt1/tyy/data/dqn_episode_dataset_pos_reward0_new.npy'
    np.save(output_file_path, replay_buffer, allow_pickle=True)
    print(f"数据集已成功保存到: {output_file_path}")


log_dir_pos = '/mnt/mnt1/tyy/data/stage3collect/'
#log_dir_pos = '/root/autodl-tmp/positive/'
#build_pos_dataset(log_dir_pos, p=0.95)


if __name__ == '__main__':
    #file_path = '/mnt/mnt1/tyy/data/stage3/stage3_collect_all.npy'
    build_pos_dataset(log_dir_pos)