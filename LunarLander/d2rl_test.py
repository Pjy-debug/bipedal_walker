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
from Env_agent.utils.criticality_model import Criticality_model_mlp,Criticality_model_trans,Reward_Model
#from Env_agent.utils.criticality_model import Criticality_model_mlp,Criticality_model_trans,sampling_trans, Reward_Model
import multiprocessing # 导入 multiprocessing 库
import os # 导入 os 库

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


def epsilon_greedy(pdf_before_epsilon, ndd_pdf, epsilon=0.1):
    # NDD epsilon greedy method
    pdf_after_epsilon = (1 - epsilon) * pdf_before_epsilon + epsilon * ndd_pdf
    #assert (0.99999 <= np.sum(pdf_after_epsilon) <= 1.0001)
    return pdf_after_epsilon  

'''
def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    max_epoch = 200000

    #model = DQN.load("Rocket_agent_withwind/model/dqn_lunar_v0.pkl",device='cpu' )
    model = DQN.load("Rocket_agent_withwind/model/dqn_lunar_v0.pkl.zip",device='cpu' )

    pre_model1 = Reward_Model()
    pre_model1.load_state_dict(torch.load('Env_agent/stage1/model/rw_model_trans_199.pt'))
    pre_model1.eval().to(device)
    # pre_model2 = Criticality_model_trans()
    # pre_model2.load_state_dict(torch.load('Env_agent/stage3/model/dqn_model_mlp_final.pt'))
    # pre_model2.eval().to(device)
    
    epsilon_model = torch.jit.load('d2rl_training/model_final.pt')
    epsilon_model.to(device)
    epsilon_model.eval()

    env = gym.make('LunarLander/ordinary_d2rl-v0', gravity = -8.5, enable_wind = True, wind_power = 10.0, turbulence_power = 1.0, criticality_model1 = pre_model1, 
    criticality_model2 = None, d2rl_model = epsilon_model, device = device)
    
    save_path = 'test_results'

    return_list = []
    episode_result = []
    failure_num = 0
    
    rewards = []
    crashes = []
    crash_num = 0
    
    for epoch in range(0,max_epoch):  
        obs,_= env.reset()
        done = False
        reward = 0
        
        total_inputs = []
        total_rewards = []
        total_actions = []
        total_dones = []
        total_all_inputs = []
        total_weights = []
        
        env_action = 0
        weight = 1
        control_step = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward_, terminated, truncated, info = env.step(action)
            done =  terminated or truncated
            crash = info['crash']
            reward+=reward_
            
            # total_inputs.append(obs)
            
            if info['criticality_info'].keys():
                all_inputs = info['criticality_info']['all_inputs']
                pdf_array = info['criticality_info']['pdf_array']
                p_list = info['criticality_info']['p_list']
                action_idx = info['criticality_info']['env_action']
                cur_weight = p_list[action_idx] / pdf_array[action_idx]
                weight *= cur_weight
                
                inputs = all_inputs[action_idx]
                if cur_weight == 1:
                    pass
                else: 
                    control_step += 1
                total_inputs.append(inputs)
                total_all_inputs.append(all_inputs)
                total_actions.append(action_idx)
                total_dones.append(0)
                total_rewards.append(0)
                total_weights.append(cur_weight)
                    
            #print(weight,action_idx)
            #print(pdf_array)
            
            if done:
                rewards.append(reward)
                crashes.append( crash * weight)
                total_dones[-1] = 1
                if crash:
                    total_rewards[-1] = 1
                    crash_num += 1
                print('d2rl epoch:', epoch, 'crash_num:', crash_num, 'weight:',weight,'all_step:',len(total_inputs),'control_step:',control_step)
        
        
        episode_info = {'total_inputs':total_inputs,
                       'total_rewards': total_rewards,
                       'total_actions':total_actions,
                       'total_dones':total_dones,
                       'failure':int(crash),
                       'total_all_inputs':total_all_inputs,
                       'total_weights':total_weights,
                       'info':info['criticality_info']}
        
        if int(crash):
            print('append failure')
            episode_result.append(episode_info)
        
                
        if epoch % 20 == 0:
            Mean, RHF, Val = calculate_val(crashes)
            print('RHF:',RHF[-1],'val:',Val[-1],'Mean:', Mean[-1])

        
        if epoch % 2000 == 0:
            print(len(crashes))
            np.save(f'{save_path}/d2rl_new_{epoch}.npy',crashes)
            episode_result = []
        if epoch % 2000 == 0:
            np.save(f'{save_path}/crash_data_d2rl_new_{epoch}.npy',episode_result)
            episode_result = []
    
    np.save(f'{save_path}/d2rl.npy',crashes)

    env.close()
'''
# 修改 main 函数，使其接受起始和结束 epoch 作为参数
def main(proc_id, start_epoch, end_epoch):
    print(f"Process {proc_id} is running from epoch {start_epoch} to {end_epoch}")
    device = torch.device(f"cuda:{proc_id % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
    
    # max_epoch = 200000
    model = DQN.load("Rocket_agent_withwind/model/dqn_lunar_v0.pkl.zip", device='cpu')
    pre_model1 = Reward_Model()
    pre_model1.load_state_dict(torch.load('Env_agent/stage1/40new_rw_model_f5_best.pt'))
    pre_model1.eval().to(device)
    epsilon_model = torch.jit.load('d2rl_training/model_final.pt')
    epsilon_model.to(device)
    epsilon_model.eval()
    env = gym.make('LunarLander/ordinary_d2rl-v0', gravity=-8.5, enable_wind=True, wind_power=10.0, turbulence_power=1.0, criticality_model1=pre_model1, 
    criticality_model2=None, d2rl_model=epsilon_model, device=device)
    
    save_path = f'test_results_new_weight/process_{proc_id}'
    os.makedirs(save_path, exist_ok=True)

    return_list = []
    episode_result = []
    failure_num = 0
    rewards = []
    crashes = []
    crash_num = 0
    
    # 循环范围改为从 start_epoch 到 end_epoch
    for epoch in range(start_epoch, end_epoch):  
        obs, _ = env.reset()
        done = False
        reward = 0
        total_inputs = []
        total_rewards = []
        total_actions = []
        total_dones = []
        total_all_inputs = []
        total_weights = []
        env_action = 0
        weight = 1
        control_step = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward_, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            crash = info['crash']
            reward += reward_
            
            if 'criticality_info' in info and info['criticality_info'].keys():
                all_inputs = info['criticality_info']['all_inputs']
                pdf_array = info['criticality_info']['pdf_array']
                p_list = info['criticality_info']['p_list']
                action_idx = info['criticality_info']['env_action']
                cur_weight = p_list[action_idx] / pdf_array[action_idx]
                weight *= cur_weight
                inputs = all_inputs[action_idx]

                
                if cur_weight != 1:
                    control_step += 1
                total_inputs.append(inputs)
                total_all_inputs.append(all_inputs)
                total_actions.append(action_idx)
                total_dones.append(0)
                total_rewards.append(0)
                total_weights.append(cur_weight)
            if weight < 1e-5:
                print(f"Process:{proc_id} Due to low weight, the episode is terminated early: {weight}")
                done = True 
                    
            if done:
                rewards.append(reward)
                crashes.append(crash * weight)
                if total_dones: # 检查列表是否为空
                    total_dones[-1] = 1
                if crash:
                    if total_rewards:
                        total_rewards[-1] = 1
                    crash_num += 1
                print(f'Process:{proc_id} d2rl epoch:{epoch} crash_num:{crash_num} weight:{weight} all_step:{len(total_inputs)} control_step:{control_step}')
        
        episode_info = {'total_inputs': total_inputs, 'total_rewards': total_rewards, 'total_actions': total_actions, 'total_dones': total_dones, 'failure': int(crash), 'total_all_inputs': total_all_inputs, 'total_weights': total_weights, 'info': info.get('criticality_info', {})}
        
        if int(crash):
            print('append failure')
            episode_result.append(episode_info)
        
        if epoch % 20 == 0:
            Mean, RHF, Val = calculate_val(crashes)
            print(f'Process:{proc_id} RHF:{RHF[-1]} val:{Val[-1]} Mean:{Mean[-1]}')
        
        if epoch % 2000 == 0:
            print(f'Process:{proc_id} has {len(crashes)} crashes')
            np.save(f'{save_path}/d2rl_new_{epoch}.npy', crashes)
            np.save(f'{save_path}/crash_data_d2rl_new_{epoch}.npy', episode_result)
            episode_result = []

    np.save(f'{save_path}/d2rl.npy', crashes)
    env.close()
# 新增的主函数，用于启动多进程
def run_multiprocess(num_processes=10, total_epochs=200000):
    processes = []
    # 计算每个进程负责的 epoch 数量
    epochs_per_proc = total_epochs // num_processes
    
    # 创建一个存放结果的根目录
    os.makedirs('test_results', exist_ok=True)
    
    for i in range(num_processes):
        start_epoch = i * epochs_per_proc
        # 最后一个进程处理剩余的所有 epoch
        end_epoch = (i + 1) * epochs_per_proc if i < num_processes - 1 else total_epochs
        
        # 创建并启动子进程
        # target=main 指定进程要运行的函数
        # args=(i, start_epoch, end_epoch) 传递给函数的参数
        p = multiprocessing.Process(target=main, args=(i, start_epoch, end_epoch))
        processes.append(p)
        p.start()
        
    # 等待所有进程完成
    for p in processes:
        p.join()

if __name__ == '__main__':
    #main()
    # 调用新的主函数来启动多进程
    run_multiprocess(num_processes=5, total_epochs=200000)

