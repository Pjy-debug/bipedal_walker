import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import gym_testenvs
from stable_baselines3 import DQN,PPO
from stable_baselines3.common.evaluation import evaluate_policy

import numpy as np
import matplotlib.pyplot as plt
import torch

import pickle
import tqdm
import math
import random
from scipy.stats import norm
from utils.criticality_model import Criticality_model_mlp,Criticality_model_trans,Reward_Model
import multiprocessing as mp
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


def epsilon_greedy(pdf_before_epsilon, ndd_pdf, epsilon=0.05):
    # NDD epsilon greedy method
    #修改，原来是（17,）无法broadcast
    ndd_pdf = ndd_pdf.reshape(-1, 1)
    pdf_after_epsilon = (1 - epsilon) * pdf_before_epsilon + epsilon * ndd_pdf
    #assert (0.99999 <= np.sum(pdf_after_epsilon) <= 1.0001)
    return pdf_after_epsilon  
# 新增的任务函数，用于每个进程独立运行
def worker_task(start_epoch, num_epochs, rank):
    # 每个进程需要重新加载模型和环境，因为它们在进程间不共享内存
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    model = DQN.load("/home/teamcommon/tyy/MyLander/Rocket_agent_withwind/model/dqn_lunar_v0.pkl", device='cpu')
    
    criticality_model = Criticality_model_trans()
    criticality_model.load_state_dict(torch.load('/home/teamcommon/tyy/MyLander/Env_agent/stage2/model/trans_49.pt'))
    criticality_model.eval().to(device)
    
    criticality_model1 = Reward_Model()
    criticality_model1.load_state_dict(torch.load('/home/teamcommon/tyy/MyLander/Env_agent/stage1/40new_rw_model_f5_best.pt'))
    criticality_model1.eval().to(device)

    env = gym.make(
        'LunarLander/ordinary_nade-v0', 
        gravity=-8.5, 
        enable_wind=True, 
        wind_power=10.0, 
        turbulence_power=1.0, 
        criticality_model1=criticality_model1, 
        criticality_model2=criticality_model
    )
    
    crashes = []
    episode_result = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        # ... (保留你原有的单个 epoch 循环逻辑) ...
        # 注意：你需要将原始 main 函数中的循环体完整复制到这里
        obs,_= env.reset()
        done = False
        reward = 0
        total_inputs = []
        total_rewards = []
        total_actions = []
        total_dones = []
        total_all_inputs = []
        env_action = 0
        weight = 1

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward_, tr, crash, info = env.step(action)
            done = crash or tr
            reward += reward_
            total_inputs.append(obs)
            
            if info['criticality_info'].keys():
                all_inputs = info['criticality_info']['all_inputs']
                pdf_array = info['criticality_info']['pdf_array']
                p_list = info['criticality_info']['p_list']
                action_idx = info['criticality_info']['env_action']
                cur_weight = p_list[action_idx] / pdf_array[action_idx]
                weight *= cur_weight
                
                inputs = all_inputs[action_idx]
                total_inputs.append(inputs)
                total_all_inputs.append(all_inputs)
                total_actions.append(action_idx)
                total_dones.append(0)
                total_rewards.append(0)
        
            if done:
                crashes.append(int(crash) * weight)
                total_dones[-1] = 1
                if int(crash):
                    total_rewards[-1] = 1
                # 打印信息加上进程号，以便区分
                print(f'rank: {rank}, epoch: {epoch}, weight: {weight}')

        episode_info = {
            'total_inputs': total_inputs,
            'total_rewards': total_rewards,
            'total_actions': total_actions,
            'total_dones': total_dones,
            'failure': int(crash),
            'total_all_inputs': total_all_inputs
        }
        if int(crash):
            episode_result.append(episode_info)
        if epoch % 5000 == 0:
            np.save(f'/mnt/mnt1/tyy/data/stage3/stage3_{rank}_collect_{epoch}.npy',episode_result)
            np.save(f'/mnt/mnt1/tyy/data/stage3/stage3_{rank}_collect_{epoch}_crashes.npy',crashes)
            episode_result = []
            print(f'rank: {rank}, saved data at epoch: {epoch}')
    env.close()
    
    # 返回每个进程的结果
    return {'crashes': crashes, 'episode_result': episode_result}
def main():
    max_epoch = 100000
    num_processes = 4  # 4个进程
    
    # 划分任务
    epochs_per_process = max_epoch // num_processes
    
    pool = mp.Pool(processes=num_processes)
    
    results = []
    for i in range(num_processes):
        start_epoch = i * epochs_per_process
        results.append(pool.apply_async(worker_task, args=(start_epoch, epochs_per_process, i)))
        
    pool.close()
    pool.join()
    
    all_crashes = []
    all_episode_result = []
    
    for res in results:
        data = res.get()
        all_crashes.extend(data['crashes'])
        all_episode_result.extend(data['episode_result'])

    # 将所有进程的结果合并后保存
    np.save('/mnt/mnt1/tyy/data/stage3/stage3_collect_nade_crashes.npy', np.array(all_crashes))
    np.save('/mnt/mnt1/tyy/data/stage3/stage3_collect_all.npy', all_episode_result)

    print(f"所有进程已完成。总共收集了 {len(all_crashes)} 个崩溃事件。")
"""
def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    max_epoch =100000

    
    model = DQN.load("/home/teamcommon/tyy/MyLander/Rocket_agent_withwind/model/dqn_lunar_v0.pkl",device='cpu')
    
    criticality_model = Criticality_model_trans()
    criticality_model.load_state_dict(torch.load('/home/teamcommon/tyy/MyLander/Env_agent/stage2/model/trans_49.pt'))
    criticality_model.eval().to(device)
    criticality_model1 = Reward_Model()
    criticality_model1.load_state_dict(torch.load('/home/teamcommon/tyy/MyLander/Env_agent/stage1/40new_rw_model_f5_best.pt'))
    criticality_model1.eval().to(device)

    env = gym.make('LunarLander/ordinary_nade-v0',gravity=-8.5,enable_wind=True,wind_power = 10.0,turbulence_power = 1.0, criticality_model1 = criticality_model1, criticality_model2 = criticality_model)
    
    return_list = []
    episode_result = []
    failure_num = 0
    crash_num = 0
    rewards = []
    crashes = []
    
    for epoch in range(0,max_epoch):  
        obs,_= env.reset()
        done = False
        reward = 0
        
        total_inputs = []
        total_rewards = []
        total_actions = []
        total_dones = []
        total_all_inputs = []
        
        env_action = 0
        weight = 1
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward_, tr, crash, info = env.step(action)
            done= crash or tr
            reward+=reward_
            
            total_inputs.append(obs)
            
            if info['criticality_info'].keys():
                all_inputs = info['criticality_info']['all_inputs']
                pdf_array = info['criticality_info']['pdf_array']
                p_list = info['criticality_info']['p_list']
                action_idx = info['criticality_info']['env_action']
                cur_weight = p_list[action_idx] / pdf_array[action_idx]
                weight *= cur_weight
                
                inputs = all_inputs[action_idx]
                total_inputs.append(inputs)
                total_all_inputs.append(all_inputs)
                total_actions.append(action_idx)
                total_dones.append(0)
                total_rewards.append(0)
            #print(weight,action_idx)
            #print(pdf_array)
        
            
            if done:
                rewards.append(reward)
                crashes.append(int(crash) * weight)
                total_dones[-1] = 1
                if int(crash):
                    total_rewards[-1] = 1
                    crash_num += 1
                print('epoch:', epoch, 'crash_num:', crash_num, 'weight:',weight)
        
        episode_info = {'total_inputs':total_inputs,
                       'total_rewards': total_rewards,
                       'total_actions':total_actions,
                       'total_dones':total_dones,
                       'failure':int(crash),
                       'total_all_inputs':total_all_inputs}
        if int(crash):
            print('append failure')
            episode_result.append(episode_info)
        
        
                
        if epoch % 5 == 0:
            Mean, RHF, Val = calculate_val(crashes)
            print('RHF:',RHF[-1],'val:',Val[-1],'Mean:',sum(crashes)/len(crashes))

        
        if epoch % 5000 == 0:
            np.save(f'/mnt/mnt1/tyy/data/stage3_collect_{epoch}.npy',episode_result)
            episode_result = []
    
    np.save('/mnt/mnt1/tyy/data/stage3_collect_nade_crashes.npy',crashes)

    env.close()
"""

if __name__ == '__main__':
    main()




