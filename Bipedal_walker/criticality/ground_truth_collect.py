import os
import time
import numpy as np
from tta.niches.box2d.model import Model

from tta.niches.box2d.env import make_env, bipedhard_custom
from tta.niches.box2d.bipedal_walker_custom import BipedalWalkerCustom, Env_config
from tta.niches.box2d.bipedal_walker_test import BipedalWalkerTest
# from criticality import calculate_criticality,calculate_val
from pjy.Bipedal_walker.criticality.not_that_important.copy_test import save_collision_step
import math
import random
from scipy.stats import norm
from criticality_mc import bipedal_walker_criticality, calu_true_criticality

# import torch

def generate_seed():
    seed = os.getpid() + int(time.time() * 1e5 % 1e6)
    return seed


alpha=0.05
z=norm.isf(q=alpha)
def calculate_val(the_list):
    Mean=[]
    Relative_half_width=[]
    Var=[]
    acc=[]
    var_old=0
    mean_old=0
    for i in range(len(the_list)):
        if math.isnan(the_list[i]) or math.isinf(the_list[i]):
            the_list[i]=0.0
        n=i+1
        mean_new=mean_old+(the_list[i]-mean_old)/n
        Mean.append(mean_new)
        var_new=(n-1)*var_old/n+(n-1)*(the_list[i]-mean_old)**2/(n*n)
        Var.append(1.96*(np.sqrt(var_new/n)))
        Relative_half_width.append(z*(np.sqrt(var_new/n)/(mean_new+1e-30)))
        var_old=var_new
        mean_old=mean_new
    return Mean,Relative_half_width,Var

log_dir = "logs/poet_new_test/"
# log_dir = "logs/poet_final_test/"
best_model_final_test = "logs/poet_final_test/poet_final_test.48542109-e29d-4fbe-8405-6618c92e990a.best.json"
best_model_new_test = "logs/poet_new_test/poet_new_test.966593ea-181f-42ec-ad1d-2ca11487bea9.best.json"
file_names = os.listdir(log_dir)
best_models = [
    log_dir + file_name for file_name in file_names if file_name.endswith('.json')]
best_models = sorted(best_models, reverse=True)
model = Model(bipedhard_custom)
get_action = model.get_action


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


ENV_CONFIG = env_config_1
RENDER_MODE = True
RGB_ARRAY = False
SAVE_REWARD = False
seed = 42
max_epoch = 1000000
best_model = best_models[39]
model.load_model(best_model)
# env = make_env("BipedalWalkerCustom", seed, RENDER_MODE, ENV_CONFIG)

env = make_env("BipedalWalkerAdv", seed, RENDER_MODE,
               ENV_CONFIG, model.get_action)
# time.sleep(3)

res = []
#res = np.load('/root/brx/tta_new/log/result_nde_100/nde_160_1290000.npy', allow_pickle=True)
#res = list(res)
failure_num = 0
for epoch in range(0,max_epoch):
    # for k in range(len(best_models)):
    observation = env.reset(seed=generate_seed())
    ternimated = False
    i = 0
    failure = 0
    is_crash = 0
    max_iters = 160
    fall_pos = 0
    weights = []
    p_list = []
    q_list = []
    episode_info = []
    trajectory = {}
    trajectory['episode'] = []
    while not ternimated and i <= max_iters:
        if i % 1000 == 0:
            print(f"epoch:{epoch},iter:{i}")
        i += 1
        ENV_ACT = env.env_act
        #print('current_i=',env.current_i,'pred_i=',env.pred_i,'agent_i=',env.agent_i)
        if ENV_ACT:
            terrain = env.terrain
            hull = env.hull
            fd_polygon = env.fd_polygon
            fd_edge = env.fd_edge
            legs = env.legs
            joints = env.joints
            drawlist = env.drawlist
            lidar = env.lidar
            vel = env.vel
            observation_space = env.observation_space
            step_info_dict = save_collision_step(env, terrain, hull, fd_polygon, fd_edge, legs, joints, drawlist, lidar, vel, observation_space)
            episode_info.append(step_info_dict)
            
            if env.terrain_state == 0:
                action = np.random.randint(5, 15)
            else:
                action = np.random.randint(5)
        else:
            action = None
        next_observation, reward, ternimated, truncated, info = env.step(action)
        
        if ENV_ACT:
          #print(i,info['agent_i'],info['position'],info['lidar_pred'])
          trajectory['episode'].append({'observation':observation, 'reward':reward, 'next_observation':next_observation,
                             'truncated':truncated,'terminated':ternimated,'env_act':action,'info':info})
        
        #print(info['cur_step_poly'])
        if ternimated and i<=max_iters:
            failure = 1
            is_crash = 1
            res.append(1)
            fall_pos = info['position'][0]
            print('failure!')
            failure_num += 1
        elif i>max_iters:
            failure = 0
            res.append(0)
    print('epoch:',epoch,'mean:',sum(res)/len(res),'failure_num:',failure_num)
    
    """
    if epoch % 10000 == 0:
        np.save(f'/root/brx/tta_new/log/result_nde_100/nde_160_{epoch}.npy',res,allow_pickle=True)
    """
    trajectory['terrain_poly'] = env.terrain_poly
    trajectory['terrain_x'] = env.terrain_x
    trajectory['terrain_y'] = env.terrain_y
    trajectory['env_actions'] = env.env_actions
    trajectory['all_pos'] = env.all_pos
    trajectory['is_failure'] = is_crash
    trajectory['fall_pos'] = fall_pos
    trajectory['failure_rate'] = []
    
    """
    if failure:
        for item in episode_info:
            failure_rate = calu_true_criticality(item,get_action)
            print(failure_rate)
            # trajectory['episode'].append(list(failure_rate))
            trajectory['failure_rate'].append(np.array(failure_rate))
        trajectory['failure_rate'] = np.array(trajectory['failure_rate'])
        np.save(f'/home/yjx/tta_new/data/mc_data/episode_{epoch}.npy', trajectory, allow_pickle=True)
    """
    if failure:
        item = episode_info[-1]
        failure_rate = calu_true_criticality(item,get_action)
        print(failure_rate)
        # trajectory['episode'].append(list(failure_rate))
        trajectory['failure_rate'].append(np.array(failure_rate))
        trajectory['failure_rate'] = np.array(trajectory['failure_rate'])
        np.save(f'/home/yjx/tta_new/data/mc_data/episode_{epoch}.npy', trajectory, allow_pickle=True)
        
        
    if epoch % 5 == 0:
        Mean, RHF, Val = calculate_val(res)
        print('RHF:',RHF[-1],'val:',Val[-1])
#np.save(f'/root/brx/tta_new/log/result_nde_100/nde_160_test.npy',res,allow_pickle=True)
Mean, RHF, Val = calculate_val(res)

env.close()

