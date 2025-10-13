'''*************************************************************************
【文件名】                 collect_data_new.py
【功能模块和目的】         收集用于训练Criticality的数据，利用BipedalWalkerAdv环境，对actions均匀采样，记录每个episode的信息
【开发者及日期】           ruoxuan | 修改: hyj | 2025
【更改记录】               last modified 2025.3.21
                        修改了文件存储位置，使得数据可以存储在指定的save_path文件夹下                
*************************************************************************'''

import sys
import os
# 获取当前文件的绝对路径
current_path = os.path.abspath(__file__)
# 向上一级目录
parent_dir = os.path.dirname(os.path.dirname(current_path))
# 将criticality目录添加到Python的模块搜索路径
sys.path.append(parent_dir)

import time
import json
import numpy as np
import tqdm
from tta.niches.box2d.model import Model
from tta.niches.box2d.env import make_env, bipedhard_custom
from tta.niches.box2d.bipedal_walker_custom import BipedalWalkerCustom, Env_config
from tta.niches.box2d.bipedal_walker_test import BipedalWalkerTest

def generate_seed():
    seed = os.getpid() + int(time.time() * 1e5 % 1e6)
    return seed

# save_path = "/mnt1/hyj/Acc_Test/tta_new/data/test_2025-6-8"
save_path = "/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new"
log_dir = "../logs/poet_new_test/"
# log_dir = "logs/poet_final_test/"
best_model_final_test = "../logs/poet_final_test/poet_final_test.48542109-e29d-4fbe-8405-6618c92e990a.best.json"
best_model_new_test = "../logs/poet_new_test/poet_new_test.966593ea-181f-42ec-ad1d-2ca11487bea9.best.json"
file_names = os.listdir(log_dir)
best_models = [
    log_dir + file_name for file_name in file_names if file_name.endswith('.json')]
best_models = sorted(best_models, reverse=True)
model = Model(bipedhard_custom)


env_config_0 = Env_config(
    name='default_env',
    ground_roughness=0,
    pit_gap=[],
    stump_width=[],
    stump_height=[],
    stump_float=[],
    stair_height=[],
    stair_width=[],
    stair_steps=[])
"""
env_config_1 = Env_config(
    name="edge",
    ground_roughness=0.6,
    pit_gap=[0, 0.8],
    stump_width=[1, 2],
    stump_height=[0.01, 0.4],
    stump_float=[0, 1],
    stair_height=[],
    stair_width=[],
    stair_steps=[])
"""
env_config_1 = Env_config(
    name="edge",
    ground_roughness=0.4,
    pit_gap=[0, 0.2],
    stump_width=[1, 2],
    stump_height=[0.01, 0.2],
    stump_float=[0, 1],
    stair_height=[],
    stair_width=[],
    stair_steps=[])

env_config_2 = Env_config(
    name="edge",
    ground_roughness=0,
    pit_gap=[2],
    stump_width=[],
    stump_height=[],
    stump_float=[],
    stair_height=[0, 2],
    stair_width=[1, 5],
    stair_steps=[1, 10])

# 应该是这个
ENV_CONFIG = env_config_1
RENDER_MODE = False # make it false to run faster 
RGB_ARRAY = False
SAVE_REWARD = False
max_epoch = 5000000
seed = 42
best_model = best_models[39]
model.load_model(best_model)
# env = make_env("BipedalWalkerCustom", seed, RENDER_MODE, ENV_CONFIG)
"""
env = make_env("BipedalWalkerTest", seed, RENDER_MODE,
               ENV_CONFIG, model.get_action)
"""

env = make_env("BipedalWalkerAdv", seed, RENDER_MODE,
               ENV_CONFIG, model.get_action)
# time.sleep(3)

cum_reward = []
frames = []

pos_raw_data = []
neg_raw_data = []
failure = 0
# for k in [40-1]:
print('collecting_data...')
for epoch in tqdm.tqdm(range(0, max_epoch)):
    # for k in range(len(best_models)):
    observation = env.reset(seed=generate_seed())
    ternimated = False
    i, r = 0, 0
    max_iters = 157
    is_crash = 0
    trajectory = {}
    trajectory['episode'] = []
    while not ternimated and i <= max_iters:
        # if i % 100 == 0:
        #     print(f"epoch:{epoch},iter:{i}")
        i += 1
        # print(f"env.terrain_state for action = {env.terrain_state}")
        # False
        ENV_ACT = env.env_act
        if ENV_ACT:
            if env.terrain_state == 0:
                action = np.random.randint(5, 15)
            else:
                action = np.random.randint(5)
        else:
            action = None
        next_observation, reward, ternimated, truncated, info = env.step(action)
        
        if ENV_ACT:
            trajectory['episode'].append({'observation':observation, 'reward':reward, 'next_observation':next_observation,
                               'truncated':truncated,'terminated':ternimated,'env_act':action,'info':info})
        # print(trajectory['episode'][-1])

        if ternimated:
            # print(info)
            if i<max_iters:
                is_crash = 1
                failure += 1
        r += reward
        """
        if ENV_ACT or i % 10 == 0:
            print(i, round(reward, 2), round(r, 2), round(
                observation[2], 2), action, round(info["position"][0], 3))
        
        if RGB_ARRAY:
            frames.append(env.render(mode="rgb_array"))
        elif RENDER_MODE:
            env.render("human")
        """
    cum_reward.append(r)
    """
    if RGB_ARRAY:
        np.save(f"data/frames_{k}", frames)
    """
    
    trajectory['terrain_poly'] = env.terrain_poly
    trajectory['terrain_x'] = env.terrain_x
    trajectory['terrain_y'] = env.terrain_y
    trajectory['env_actions'] = env.env_actions
    trajectory['all_pos'] = env.all_pos
    trajectory['is_failure'] = is_crash
    #print(trajectory)

    """
    print(len(env.terrain_poly),env.terrain_poly)
    print(len(env.terrain_x),env.terrain_x)
    print(len(env.terrain_y),env.terrain_y)
    print(len(env.env_actions), env.env_actions)
    print(len(env.all_pos), env.all_pos)
    for item in env.env_actions:
        print(item)
    """
    if is_crash:
        pos_raw_data.append(trajectory)
        # np.save(f'/mnt/new_raw_data/pos/crash/crash_{epoch}.npy',[trajectory],allow_pickle=True)
        # 检测路径是否存在
        if not os.path.exists(save_path+'/raw_data/pos/crash'):
            os.makedirs(save_path+f'/raw_data/pos/crash')
        np.save(save_path+f'/raw_data/pos/crash/crash_{epoch}.npy',[trajectory],allow_pickle=True)
    else:
        neg_raw_data.append(trajectory)
        #np.save(f'/mnt/new_raw_data/neg/crash/crash_{epoch}.npy',[trajectory],allow_pickle=True)
    
    # print('failure_num:', failure)
    if len(pos_raw_data) == 20:
        """
        with open(f'data/raw_data/transitions_{epoch//1000}.json', 'w', encoding='utf8') as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)
        """
        # np.save(f'/mnt/new_raw_data/pos/transitions_{epoch}.npy',pos_raw_data,allow_pickle=True)
        if not os.path.exists(save_path+'/raw_data/pos'):
            os.makedirs(save_path+'/raw_data/pos')
        np.save(save_path+f'/raw_data/pos/transitions_{epoch}.npy',pos_raw_data,allow_pickle=True)
        pos_raw_data = []
        
    if len(neg_raw_data) == 2000:
        """
        with open(f'data/raw_data/transitions_{epoch//1000}.json', 'w', encoding='utf8') as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)
        """
        if not os.path.exists(save_path+'/raw_data/neg'):
            os.makedirs(save_path+'/raw_data/neg')
        np.save(save_path+f'/raw_data/neg/transitions_{epoch}.npy',neg_raw_data,allow_pickle=True)
        neg_raw_data = []

env.close()

if SAVE_REWARD:
    if not os.path.exists(save_path+'/reward_data'):
        os.makedirs(save_path+'/reward_data')
    np.save(save_path+f'/reward_data/cum_reward.csv', cum_reward, delimiter=",", fmt="%.4f")
        
