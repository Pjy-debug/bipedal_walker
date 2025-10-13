'''*************************************************************************
[file name]                 nade_test.py
[description]              nade_test.py与d2rl_test.py是加速测试环境，其中criticality计算分别在
                            criticality_new.py与criticality_d2rl.py中
[developer]                brx, 2024
[changelog]               （若修改过则必需注明）
[usage]                    
*************************************************************************'''
import sys
import os
# 获取当前文件的绝对路径
current_path = os.path.abspath(__file__)
# 向上一级目录
parent_dir = os.path.dirname(os.path.dirname(current_path))
# 将criticality目录添加到Python的模块搜索路径
sys.path.append(parent_dir)


import os
import time
import numpy as np
from tta.niches.box2d.model import Model
import tqdm

from tta.niches.box2d.env import make_env, bipedhard_custom
from tta.niches.box2d.bipedal_walker_custom import BipedalWalkerCustom, Env_config
from tta.niches.box2d.bipedal_walker_test import BipedalWalkerTest
from criticality_.criticality_new import calculate_criticality,calculate_val
import matplotlib.pyplot as plt
from matplotlib import animation

import torch
# import pyglet
# pyglet.options["headless"] = True

log_dir = "../logs/poet_new_test/"
# log_dir = "logs/poet_final_test/"
best_model_final_test = "../logs/poet_final_test/poet_final_test.48542109-e29d-4fbe-8405-6618c92e990a.best.json"
best_model_new_test = "../logs/poet_new_test/poet_new_test.966593ea-181f-42ec-ad1d-2ca11487bea9.best.json"
file_names = os.listdir(log_dir)
best_models = [
    log_dir + file_name for file_name in file_names if file_name.endswith('.json')]
best_models = sorted(best_models, reverse=True)
model = Model(bipedhard_custom)

def display_frames_as_gif(frames,epoch):
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval = 5)
    #anim.save(f"crash_data/frames/crash/test_result_{epoch}.gif", writer="pillow", fps = 1000)
    
    anim.save(f"crash_data/render/render_{epoch}.gif", writer="pillow", fps = 1000)


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
RGB_ARRAY = False # False
SAVE_REWARD = False
experiment_name = "nade_test_2025-10-13"
# res_save_path = f'/mnt1/hyj/Acc_Test/tta_new/tta/data/{experiment_name}'
res_save_path = f'my_test/{experiment_name}'
if not os.path.exists(res_save_path):
    os.makedirs(res_save_path)
seed = 42
max_epoch = 1500000
best_model = best_models[39]
model.load_model(best_model)
# env = make_env("BipedalWalkerCustom", seed, RENDER_MODE, ENV_CONFIG)

env = make_env("BipedalWalkerAdv", seed, RENDER_MODE,
               ENV_CONFIG, model.get_action)
# time.sleep(3)

# res = []
# new_data_2_330000
res_mean = []
res_mean_part = []
#res_mean = np.load('/root/brx/tta_new/log/result_nde_100/new_data_160_476_1100000.npy', allow_pickle=True)
#res_mean = list(res_mean)
failure_num = 0
weights_2 = []
res_mean_2 = []
for epoch in tqdm.tqdm(range(0,500000)):
    # for k in range(len(best_models)):
    frames = []
    observation = env.reset(seed=epoch+100)
    ternimated = False
    i = 0
    failure = 0
    max_iters = 157 # 160
    weights = []
    p_list = []
    q_list = []
    while not ternimated and i <= max_iters:
        img = env.render(mode='rgb_array', close=True)
        frames.append(img)
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
            weights.append(cur_weight)
            p_list.append(p_orin)
            q_list.append(q)
            #weights_2.append(weight_2)
            #print(cur_weight,p_orin,q)
            #print('my env step poly:',poly)
            
        else:
            cur_action = None
            
        next_observation, reward, ternimated, truncated, info,  = env.step(cur_action)
        cur_step_poly = info['cur_step_poly']
        #print('env poly:',cur_step_poly)
        
        if ternimated and i<=max_iters:
            failure = 1
            failure_num += 1
            res_mean.append(np.prod(weights))
            res_mean_part.append(np.prod(weights))
            """
            res.append({'weights':weights,'p_list':p_list,'q_list':q_list,'failure':failure})
            res_mean.append(np.prod(weights))
            res_mean_2.append(np.prod(weights_2))
            """
        elif i>max_iters:
            failure = 0
            res_mean.append(0)
            res_mean_part.append(0)
            """
            res.append({'weights':weights,'p_list':p_list,'q_list':q_list,'failure':failure})
            res_mean.append(0)
            res_mean_2.append(0)
            """
    # print('epoch:',epoch,'mean:',sum(res_mean)/len(res_mean),'failure_num:',failure_num,'failure:',failure,'weights:',np.prod(weights))
    
    # np.save(f'/home/ubuntu/tta_new/tta/data/render/render_{epoch}.npy',frames)
    if RGB_ARRAY: 
        # save_path = f'/mnt1/hyj/Acc_Test/tta_new/tta/data/render/{experiment_name}'
        save_path = f'my_test/render/{experiment_name}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(save_path + f'/render_{epoch}.npy', frames)
    
    if epoch % 1000 == 0:
        np.save(res_save_path + f'/nade_160_test_{epoch}.npy',res_mean_part,allow_pickle=True)
        res_mean_part = []
        #np.save(f'/root/brx/tta_new/log/result_nde_100/nade_{epoch}.npy',res,allow_pickle=True)
        # new_data,1,2,3
        # np.save(f'/root/brx/tta_new/log/result_nde_100/new_data_160_546_rescale_{epoch}.npy',res_mean,allow_pickle=True)
    
    # if epoch % 5 == 0:
    #     Mean, RHF, Val = calculate_val(res_mean)
    #     print('RHF:',RHF[-1],'val:',Val[-1])
    
#np.save(f'/root/brx/tta_new/log/result_nde_100/nade_test.npy',res,allow_pickle=True)
# new_data_1
# np.save(f'/root/brx/tta_new/log/result_nde_100/new_data_160_546_rescale.npy',res_mean,allow_pickle=True)

np.save(res_save_path + f'/nade_160_test.npy',res_mean,allow_pickle=True)
Mean, RHF, Val = calculate_val(res_mean)
print('Mean:', Mean)
print('Final RHF:', RHF[-1], 'Final Val:', Val[-1])

env.close()
