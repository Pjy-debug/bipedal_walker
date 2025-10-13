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

from Env_agent.utils.criticality_model import Criticality_model_mlp,Criticality_model_trans, sampling_trans, Reward_Model



import matplotlib.pyplot as plt
from matplotlib import animation

def display_frames_as_gif(frames,epoch):
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval = 5)
    anim.save(f"crash_data/test_result_{epoch}.gif", writer="pillow", fps = 1000)

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
    
def main():
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    max_epoch =10

    
    model = DQN.load("Rocket_agent_withwind/model/dqn_lunar_v0.pkl",device=device)
    
    pre_model1 = Reward_Model()
    pre_model1.load_state_dict(torch.load("/mnt1/brx/Rocketdata/model/stage1/rw_model_f5_99.pt"))
    pre_model1.eval().to(device)
    pre_model2 = Criticality_model_trans()
    pre_model2.load_state_dict(torch.load("/mnt1/brx/Rocketdata/model/stage3/dqn_model_50.pt"))
    
    pre_model2.eval().to(device)
    
    env = gym.make('LunarLander/ordinary_nade-v0',render_mode='rgb_array', gravity=-8.5,enable_wind=True,wind_power = 10.0,turbulence_power = 1.0, criticality_model1 = pre_model1,criticality_model2 = pre_model2, device=device)
    


    return_list = []
    episode_result = []
    failure_num = 0
    
    rewards = []
    crashes = []
    crash_num = 0
    #crashes = np.load('crash_data/test_result/nade_f5_wo1_240000.npy',allow_pickle=True)
    #crash_num = (crashes>0).sum()
    #crashes = list(crashes)
    
    for epoch in range(1):  
        obs,_= env.reset()
        done = False
        reward = 0
        frames = []
        
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
            frames.append(env.render())
            action, _states = model.predict(obs, deterministic=True)
            obs, reward_, terminated, truncated, info = env.step(action)
            done =  terminated or truncated
            crash = info['crash']
            #crash = int(truncated)
            reward+=reward_
            
            
            # total_inputs.append(obs)
            """
            criticality = []
            p_list = []
            all_inputs = []
            for k in range(17):
                cur_input = list(obs) + [info['wind_mag'],info['torque_mag'],k]
                cur_input = torch.tensor(cur_input, dtype=torch.float32,device=device)
                inputs = cur_input.reshape(1,-1)
                outputs,_,_,_ = criticality_model(inputs,inputs,0.5)
                #print(outputs[0])
                q = outputs[0][-1].item()
                criticality.append(q)
                p_list.append(1)
                all_inputs.append(cur_input)
            if sum(criticality):
                criticality = np.array(criticality) / sum(criticality)
                p_list = np.array(p_list) / sum(p_list)
                pdf_array = epsilon_greedy(criticality, p_list) 
            else:
                p_list = np.array(p_list) / sum(p_list)
                pdf_array = p_list
            
            action_idx = np.random.choice(17, 1, p=pdf_array)
            action_idx = action_idx[0]
            """
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
                if crash and weight > 1:
                    continue
                rewards.append(reward)
                crashes.append( crash * weight)
                total_dones[-1] = 1
                if crash:
                    total_rewards[-1] = 1
                    crash_num += 1
                print('epoch:', epoch, 'crash_num:', crash_num, 'weight:',weight,'all_step:',len(total_inputs),'control_step:',control_step)
        
        
        episode_info = {'total_inputs':total_inputs,
                       'total_rewards': total_rewards,
                       'total_actions':total_actions,
                       'total_dones':total_dones,
                       'failure':int(crash),
                       'total_all_inputs':total_all_inputs,
                       'total_weights':total_weights,
                       'frames':frames}
        
        if not crash:
            display_frames_as_gif(frames,epoch)
            np.save(f'crash_data/render/render_{epoch}.npy',[episode_info])
        
        if int(crash):
            print('append failure')
            episode_result.append(episode_info)
        
        
                
        if epoch % 20 == 0:
            Mean, RHF, Val = calculate_val(crashes)
            print('09 RHF:',RHF[-1],'val:',Val[-1],'Mean:', Mean[-1])

        
        if epoch % 2000 == 0:
            print(len(crashes))
            np.save(f'/home/yjx/MyLander/crash_data/test_result_05/nade_render_{epoch}.npy',crashes)

        if epoch % 2000 == 0:
            #np.save(f'/home/yjx/MyLander/crash_data/test_result/crash_data_render_{epoch}.npy',episode_result)
            episode_result = []

    
    np.save('/home/yjx/MyLander/crash_data/test_result/nade_result_1_new.npy',crashes)

    env.close()


if __name__ == '__main__':
    main()
