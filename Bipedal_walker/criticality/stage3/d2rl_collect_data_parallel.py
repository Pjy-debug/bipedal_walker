import os
import time
import numpy as np
from tta.niches.box2d.model import Model
import json

from tta.niches.box2d.env import make_env, bipedhard_custom
from tta.niches.box2d.bipedal_walker_custom import BipedalWalkerCustom, Env_config
from tta.niches.box2d.bipedal_walker_test import BipedalWalkerTest
from criticality_new import calculate_criticality, calculate_val

# from utils.seeding import generate_seed
import datetime
import multiprocessing as mp
from multiprocessing import Process
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch

def generate_seed():
    seed = os.getpid() + int(time.time() * 1e5 % 1e6)
    return seed

def nade_test(k, env_config, model):
    failure_num = 0
    trange = tqdm(range((k+10)*100000,(k+11)*100000))
    
    
    RENDER_MODE = False
    RGB_ARRAY = False
    SAVE_REWARD = False
    seed = 42

    env = make_env("BipedalWalkerAdv", seed, RENDER_MODE, env_config, model.get_action)

    for epoch in trange:
        # 一个episode
        episode_data = {}
        observation = env.reset(seed=generate_seed())
        ternimated = False
        i = 0
        failure = 0
        max_iters = 160
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
            """
            if i % 1000 == 0:
                print(f"epoch:{epoch},iter:{i}")
            """
            i += 1
            ENV_ACT = env.env_act
            if ENV_ACT:
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
                
                weight_step_info[f'{i}'] = cur_weight
                drl_epsilon_step_info[f'{i}'] = 0.925
                real_epsilon_step_info[f'{i}'] = 0.925
                q_step_info[f'{i}'] = q
                ndd_step_info[f'{i}'] = p_orin
                criticality_step_info[f'{i}'] = c

                weights.append(cur_weight)
                p_list.append(p_orin)
                q_list.append(q)
                # weights_2.append(weight_2)
                # print(cur_weight,p_orin,q)
                # print('my env step poly:',poly)

            else:
                cur_action = None
                criticality_step_info[f'{i}'] = 0.0

            next_observation, reward, ternimated, truncated, info, = env.step(cur_action)
            cur_step_poly = info['cur_step_poly']
            if ENV_ACT:
                drl_obs_step_info[f'{i}'] = next_observation.tolist()

            if ternimated and i <= max_iters:
                failure_num += 1
                is_failure = True
                end_time = i

            elif i > max_iters:
                end_time = max_iters

        #print('epoch:', epoch, 'failure_num:', failure_num, 'failure:', is_failure, 'weights:', np.prod(weights))
        if is_failure:
            episode_data['weight_step_info '] = weight_step_info
            episode_data['drl_epsilon_step_info '] = drl_epsilon_step_info
            episode_data['real_epsilon_step_info'] = real_epsilon_step_info
            episode_data['criticality_step_info'] = criticality_step_info
            episode_data['ndd_step_info'] = ndd_step_info
            episode_data['q_step_info'] = q_step_info
            episode_data['drl_obs_step_info'] = drl_obs_step_info
            episode_data['weight_episode'] = np.prod(weights)
            episode_data['episdoe_info'] = {'id': epoch, 'start_time': 0, 'end_time': end_time}

            file_name = f'/root/autodl-tmp/data/crash/{epoch}.json'
            with open(file_name, 'w') as f:
                json.dump(episode_data, f)
            #print(f'save json file {file_name}!')

        env.close()
    return 0

if __name__ == '__main__':

    start_t = datetime.datetime.now()
    num_cores = int(mp.cpu_count())
    print("total available threads = " + str(num_cores))
    torch.multiprocessing.set_start_method('spawn')
    
    log_dir = "logs/poet_new_test/"
    # log_dir = "logs/poet_final_test/"
    best_model_final_test = "logs/poet_final_test/poet_final_test.48542109-e29d-4fbe-8405-6618c92e990a.best.json"
    best_model_new_test = "logs/poet_new_test/poet_new_test.966593ea-181f-42ec-ad1d-2ca11487bea9.best.json"
    file_names = os.listdir(log_dir)
    best_models = [
        log_dir + file_name for file_name in file_names if file_name.endswith('.json')]
    best_models = sorted(best_models, reverse=True)
    model = Model(bipedhard_custom)
    best_model = best_models[39]
    model.load_model(best_model)

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
    #ENV_CONFIG = env_config_1
    
    """
    num_cores_use = 2
    num_test_per_core = 10000000 // num_cores_use
    pool = mp.Pool(num_cores_use)
    """
    process_list = []
    
    
    
    #results = [pool.apply_async(nade_test, args=(num,env_config_1)) for num in np.ones(num_cores_use) * num_test_per_core]
    #episode_rewards = np.array([p.get()[0] for p in results])
    #positions = np.array([p.get()[1] for p in results])
    
    for k in range(20):
        print(k)
        #pool.apply_async(nade_test, args=(k,env_config_1,model,))
        #nade_test(k,env_config_1,model)
        p = Process(target=nade_test,args=(k,env_config_1,model,)) #实例化进程对象
        p.start()
        process_list.append(p)

    #pool.close()
    #pool.join()
    for i in process_list:
        p.join()

    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    #np.save('data/rewards_1e7_0', episode_rewards.flatten())
    #np.save('/root/autodl-tmp/data/positions_1e7_0', positions.flatten())
    print("total time spent = " + "{:.2f}".format(elapsed_sec / 60) + " min")