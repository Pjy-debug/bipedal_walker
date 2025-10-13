'''*************************************************************************
【文件名】                 d2rl_collect_data.py
【功能模块和目的】         收集json数据，利用BipedalWalkerAdv环境，对actions均匀采样，记录每个episode的信息
【开发者及日期】           ruoxuan | 修改: hyj | 2025
【更改记录】               last modified 2025.3.21
                        修改了文件存储位置，使得数据可以存储在指定的save_path文件夹下                
*************************************************************************'''
import os
import time
import numpy as np
from tta.niches.box2d.model import Model
import json

from tta.niches.box2d.env import make_env, bipedhard_custom
from tta.niches.box2d.bipedal_walker_custom import BipedalWalkerCustom, Env_config
from tta.niches.box2d.bipedal_walker_test import BipedalWalkerTest
from criticality_new import calculate_criticality,calculate_val

import torch

save_path = "/mnt1/hyj/Acc_Test/tta_new/data/d2rl_2025-6-8"
log_dir = "logs/poet_new_test/"
# log_dir = "logs/poet_final_test/"
best_model_final_test = "logs/poet_final_test/poet_final_test.48542109-e29d-4fbe-8405-6618c92e990a.best.json"
best_model_new_test = "logs/poet_new_test/poet_new_test.966593ea-181f-42ec-ad1d-2ca11487bea9.best.json"
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
RENDER_MODE = True
RGB_ARRAY = False
SAVE_REWARD = False
seed = 42
max_epoch = 5000000
best_model = best_models[39]
model.load_model(best_model)
# env = make_env("BipedalWalkerCustom", seed, RENDER_MODE, ENV_CONFIG)

env = make_env("BipedalWalkerAdv", seed, RENDER_MODE,
               ENV_CONFIG, model.get_action)
# time.sleep(3)

res = []
res_mean = []
failure_num = 0
weights_2 = []
res_mean_2 = []

for epoch in range(0,max_epoch):
    # 一个episode
    episode_data = {}
    observation = env.reset(seed=epoch)
    ternimated = False
    i = 0
    failure = 0
    # max_iters = 157
    max_iters = 1300
    #max_iters = 160
    weights = []
    p_list = []
    q_list = []
    episode_data['initial_obs'] = observation.tolist()
    episode_data['initial_criticality'] = 1
    episode_data['initial_weight'] = 1
    
    weight_step_info = {}
    drl_epsilon_step_info = {}
    real_epsilon_step_info = {}
    criticality_step_info = {}
    ndd_step_info = {}
    q_step_info = {}
    drl_obs_step_info = {}
    
    is_failure = False
    end_time = 0
    
    while not ternimated and i <= max_iters:
        # if i % 1000 == 0:
        #     print(f"epoch:{epoch},iter:{i}")
        i += 1
        ENV_ACT = env.env_act
        #print('current_i=',env.current_i,'pred_i=',env.pred_i,'agent_i=',env.agent_i)
        if ENV_ACT:
            """
            if env.terrain_state == 0:
                cur_action = np.random.randint(5, 15)
            else:
                cur_action = np.random.randint(5)
            #print('before:',env.state)
            """
            
            cur_action, cur_weight, p_orin, q, c = calculate_criticality(env.terrain_state,
                                                                      env.state,
                                                                      env.pos, 
                                                                      env.terrain, 
                                                                      env.current_i,
                                                                      env.agent_i, 
                                                                      env.pred_i, 
                                                                      env.terrain_counter, 
                                                                      env.terrain_x, 
                                                                      env.terrain_velocity, 
                                                                      env.current_y, 
                                                                      env.env_params, 
                                                                      env.config, 
                                                                      env.env_actions,
                                                                      env.np_random)
            #print('after:',env.current_i,env.terrain_state,env.pos,env.agent_i,env.pred_i,env.np_random,env.terrain_counter,env.terrain_velocity)
            weight_step_info[f'{i}'] = cur_weight
            drl_epsilon_step_info[f'{i}'] = 0.95
            real_epsilon_step_info[f'{i}'] = 0.95
            q_step_info[f'{i}'] = q
            ndd_step_info[f'{i}'] = p_orin
            criticality_step_info[f'{i}'] = c
            
            weights.append(cur_weight)
            p_list.append(p_orin)
            q_list.append(q)
            #weights_2.append(weight_2)
            #print(cur_weight,p_orin,q)
            #print('my env step poly:',poly)
            
        else:
            cur_action = None
            criticality_step_info[f'{i}'] = 0.0
            
        next_observation, reward, ternimated, truncated, info,  = env.step(cur_action)
        cur_step_poly = info['cur_step_poly']
        #print('env poly:',cur_step_poly)
        if ENV_ACT:
            drl_obs_step_info[f'{i}'] = next_observation.tolist()
        
        if ternimated and i<=max_iters:
            failure_num += 1
            is_failure = True
            end_time = i
            
        elif i>max_iters:
            end_time = max_iters
                
    print('epoch:',epoch,'failure_num:',failure_num,'failure:',is_failure,'weights:',np.prod(weights), 'steps:',i)
    if is_failure:
        episode_data['weight_step_info '] = weight_step_info 
        episode_data['drl_epsilon_step_info '] = drl_epsilon_step_info 
        episode_data['real_epsilon_step_info'] = real_epsilon_step_info 
        episode_data['criticality_step_info'] = criticality_step_info 
        episode_data['ndd_step_info'] = ndd_step_info 
        episode_data['q_step_info'] = q_step_info
        episode_data['drl_obs_step_info'] = drl_obs_step_info
        episode_data['weight_episode'] = np.prod(weights)
        episode_data['episdoe_info'] = {'id': epoch, 'start_time':0, 'end_time':end_time}
        
        # file_name = f'/home/ubuntu/tta_new/tta/data/d2rl/crash_{epoch}.json'
        file_name = save_path + f'/crash_{epoch}.json'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(file_name,'w') as f:
            json.dump(episode_data,f, indent=4)
        print(f'save json file {file_name}!')
    
    
    """
    if epoch % 100000 == 0:
        #np.save(f'/root/brx/tta_new/log/result_nde_100/nade_{epoch}.npy',res,allow_pickle=True)
        # new_data,1,2,3
        np.save(f'/root/brx/tta_new/log/result_nde_100/new_data_160_546_{epoch}.npy',res_mean,allow_pickle=True)
    
    if epoch % 5 == 0:
        Mean, RHF, Val = calculate_val(res_mean)
        print('RHF:',RHF[-1],'val:',Val[-1])
    """
env.close()

