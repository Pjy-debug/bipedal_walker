import os
import time
import numpy as np
from tta.niches.box2d.model import Model

from tta.niches.box2d.env import make_env, bipedhard_custom
from tta.niches.box2d.bipedal_walker_custom import BipedalWalkerCustom, Env_config
from tta.niches.box2d.bipedal_walker_test import BipedalWalkerTest
#from criticality_new import calculate_criticality, calculate_val
from criticality_mc import bipedal_walker_criticality,calculate_val



def epsilon_greedy(pdf_before_epsilon, ndd_pdf, epsilon=0.05):
    # NDD epsilon greedy method
    pdf_after_epsilon = (1 - epsilon) * pdf_before_epsilon + epsilon * ndd_pdf
    #assert (0.99999 <= np.sum(pdf_after_epsilon) <= 1.0001)
    return pdf_after_epsilon

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

# 应该是这个
ENV_CONFIG = env_config_1
RENDER_MODE = True
RGB_ARRAY = False
SAVE_REWARD = False
seed = 42
max_epoch = 1500000
best_model = best_models[39]
model.load_model(best_model)
# env = make_env("BipedalWalkerCustom", seed, RENDER_MODE, ENV_CONFIG)
get_action = model.get_action
env = make_env("BipedalWalkerAdv", seed, RENDER_MODE,
               ENV_CONFIG, model.get_action)
# time.sleep(3)

res = []
# new_data_2_330000
res_mean = []
# res_mean = np.load('/root/brx/tta_new/log/result_nde_100/new_data_160_476_1100000.npy', allow_pickle=True)
# res_mean = list(res_mean)
failure_num = 0
weights_2 = []
res_mean_2 = []
failures_episode = []
raw_data = []
for epoch in range(2000001, 2000001 + max_epoch):
    env = make_env("BipedalWalkerAdv", seed, RENDER_MODE,
               ENV_CONFIG, model.get_action)
    # for k in range(len(best_models)):
    observation = env.reset(seed=epoch + 100)
    ternimated = False
    i = 0
    failure = 0
    max_iters = 160
    weights = []
    p_list = []
    q_list = []
    failure_episode = []
    trajectory = {}
    trajectory['episode'] = []
    while not ternimated and i <= max_iters:
        if i % 1000 == 0:
            print(f"epoch:{epoch},iter:{i}")
        i += 1
        ENV_ACT = env.env_act
        terrain_state = env.terrain_state
        # print('current_i=',env.current_i,'pred_i=',env.pred_i,'agent_i=',env.agent_i)
        if ENV_ACT:
            """
            if env.terrain_state == 0:
                cur_action = np.random.randint(5, 15)
            else:
                cur_action = np.random.randint(5)
            #print('before:',env.state)
            """
            failure_rate = bipedal_walker_criticality(env,get_action)
            print(failure_rate)
            trajectory['episode'].append(list(failure_rate))

            if terrain_state == 0:
                p_list = [0.1] * 10
                p_list = np.array(p_list)
                if sum(failure_rate)>0:
                    criticality = np.array(failure_rate) / sum(failure_rate)
                    pdf_array = epsilon_greedy(criticality, p_list, epsilon=0.05)
                else:
                    pdf_array = p_list
                action_idx = np.random.choice(10, 1, p=pdf_array)
                action_idx = action_idx[0]
                cur_weight = p_list[action_idx] / pdf_array[action_idx]
                cur_action = action_idx + 5
                
            else:
                p_list = [0.2] * 5
                p_list = np.array(p_list)
                if sum(failure_rate)>0:
                    criticality = np.array(failure_rate) / sum(failure_rate)
                    pdf_array = epsilon_greedy(criticality, p_list, epsilon=0.05)
                else:
                    pdf_array = p_list
                action_idx = np.random.choice(5, 1, p=pdf_array)
                action_idx = action_idx[0]
                cur_weight = p_list[action_idx] / pdf_array[action_idx]
                cur_action = action_idx
            
            weights.append(cur_weight)

        else:
            cur_action = None

        next_observation, reward, ternimated, truncated, info, = env.step(cur_action)
        cur_step_poly = info['cur_step_poly']
        # print('env poly:',cur_step_poly)

        if ternimated and i <= max_iters:
            failure = 1
            failure_num += 1
            is_crash = 1
            res_mean.append(np.prod(weights))
            fall_pos = info['position'][0]
           
            
        elif i > max_iters:
            failure = 0
            is_crash = 0
            res_mean.append(0)
            fall_pos = 0
    
    trajectory['terrain_poly'] = env.terrain_poly
    trajectory['terrain_x'] = env.terrain_x
    trajectory['terrain_y'] = env.terrain_y
    trajectory['env_actions'] = env.env_actions
    trajectory['all_pos'] = env.all_pos
    trajectory['is_failure'] = is_crash
    trajectory['fall_pos'] = fall_pos
        
    raw_data.append(trajectory)   
           
    print('epoch:', epoch, 'mean:', sum(res_mean) / len(res_mean), 'failure_num:', failure_num, 'failure:', failure,
          'weights:', np.prod(weights))

    if epoch % 10 == 0:
        print(failures_episode)
        np.save(f'/home/yjx/tta_new/result/nade/nade_mc_{epoch}.npy',raw_data,allow_pickle=True)
        raw_data = []
        # new_data,1,2,3
        """
        np.save(f'/home/yjx/tta_new/log/result_nde_100/new_data_160_546_rescale_{epoch}.npy', res_mean,
                allow_pickle=True)
        """

    if epoch % 5 == 0:
        Mean, RHF, Val = calculate_val(res_mean)
        print('RHF:', RHF[-1], 'val:', Val[-1])

# np.save(f'/root/brx/tta_new/log/result_nde_100/nade_test.npy',res,allow_pickle=True)
# new_data_1
#np.save(f'/home/yjx/tta_new/result/nade/nade_mc.npy', res_mean, allow_pickle=True)
Mean, RHF, Val = calculate_val(res_mean)

env.close()

