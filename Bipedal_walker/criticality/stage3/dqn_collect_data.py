'''*************************************************************************
【文件名】                 dqn_collect_data.py
【功能模块和目的】         利用BipedalWalkerAdv环境，利用DQN提供actions，记录每个episode的信息
【开发者及日期】           ruoxuan | 修改: hyj | 2025
【更改记录】               last modified 2025.3.21
                        修改了文件存储位置，使得数据可以存储在指定的save_path文件夹下                
*************************************************************************'''
import os
import time
import json
import numpy as np
from tta.niches.box2d.model import Model
from tta.niches.box2d.env import make_env, bipedhard_custom
from tta.niches.box2d.bipedal_walker_custom import BipedalWalkerCustom, Env_config
from tta.niches.box2d.bipedal_walker_test import BipedalWalkerTest
from criticality_new import calculate_criticality,calculate_val, dqn_collect_data

save_path = "/mnt1/hyj/Acc_Test/tta_new/data/dqn_neg_raw_data"
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
max_epoch = 100000
seed = 42
best_model = best_models[39]  # 37
model.load_model(best_model)
# env = make_env("BipedalWalkerCustom", seed, RENDER_MODE, ENV_CONFIG)
env = make_env("BipedalWalkerAdv", seed, RENDER_MODE,
               ENV_CONFIG, model.get_action)
# time.sleep(3)

cum_reward = []
frames = []


def generate_seed():
    seed = os.getpid() + int(time.time() * 1e5 % 1e6)
    return seed

raw_data = []
failure = 0

# for k in [40-1]:
# for epoch in range(max_epoch+1,500000):
for epoch in range(max_epoch):
    # for k in range(len(best_models)):
    observation = env.reset(seed=generate_seed())
    ternimated = False
    is_crash = 0
    fall_pos = 0
    i, r = 0, 0
    max_iters = 2000 # 160
    trajectory = {}
    trajectory['episode'] = []
    total_inputs = []
    total_all_inputs = []
    total_rewards = []
    total_dones = []
    total_terrain_states = []
    total_actions = []
    while not ternimated and i <= max_iters:
        if i % 200 == 0:
            print(f"epoch:{epoch},iter:{i}")
        i += 1
        # print(f"env.terrain_state for action = {env.terrain_state}")
        # False
        ENV_ACT = env.env_act
        if ENV_ACT:
            """
            if env.terrain_state == 0:
                action = np.random.randint(5, 15)
            else:
                action = np.random.randint(5)
            """
            cur_action, cur_inputs, all_inputs = dqn_collect_data(env.terrain_state,
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
            total_inputs.append(cur_inputs)
            total_all_inputs.append(all_inputs)
            total_rewards.append(0)
            total_actions.append(cur_action)
            total_terrain_states.append(env.terrain_state)
        else:
            cur_action = None
        next_observation, reward, ternimated, truncated, info = env.step(cur_action)
        
        if ENV_ACT:
          #print(i,info['agent_i'],info['position'],info['lidar_pred'])
          if ternimated:
              total_dones.append(1)
          else:
              total_dones.append(0)
          trajectory['episode'].append({'observation':observation, 'reward':reward, 'next_observation':next_observation,
                             'truncated':truncated,'terminated':ternimated,'env_act':cur_action,'info':info})
        # print(trajectory['episode'][-1])

        if ternimated and i<max_iters:
          failure += 1
          is_crash = 1
          fall_pos = info['position'][0]
          total_rewards[-1] = 1
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
    trajectory['fall_pos'] = fall_pos
    
    trajectory['inputs'] = total_inputs
    trajectory['all_inputs'] = total_all_inputs
    trajectory['rewards'] = total_rewards
    trajectory['dones'] = total_dones
    trajectory['actions'] = total_actions
    trajectory['terrain_states'] = total_terrain_states

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
    if not is_crash:
        raw_data.append(trajectory)
    #print(trajectory['episode'])
    print('failure_num:', failure)
    if epoch % 5000 == 0:
        """
        with open(f'data/raw_data/transitions_{epoch//1000}.json', 'w', encoding='utf8') as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)
        """
        #np.save(f'/home/yjx/tta_new/data/raw_data_dqn/transitions_{epoch//10}.npy',raw_data,allow_pickle=True)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(save_path+f'/transitions_{epoch//10}.npy',raw_data,allow_pickle=True)
        raw_data = []

env.close()

if SAVE_REWARD:
    np.savetxt("data/cum_reward.csv", cum_reward, delimiter=",", fmt="%.4f")
