from stable_baselines3 import DQN,PPO
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt
import torch

import pickle
import tqdm
import math
import random
from scipy.stats import norm
from Env_agent.utils.criticality_model import Criticality_model_mlp,Criticality_model_trans, Reward_Model

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import gymnasium as gym
import gym_testenvs
from stable_baselines3 import DQN,PPO
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt
import torch
import argparse 

import pickle
import tqdm
import math
import random
from scipy.stats import norm
from Env_agent.utils.criticality_model import Criticality_model_mlp,Criticality_model_trans, sampling_trans, Reward_Model

import os
import time
import numpy as np
import json

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj



class FCNorm(nn.Module):
    def __init__(self, num_features, num_classes):
        super(FCNorm, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.linear(F.normalize(x), F.normalize(self.weight))
        return out

class distill_q(nn.Module):
    def __init__(self,input_dim=10, embed_dim_1=64, embed_dim_2=128, num_classes=17, dropout=0.1):
        super(distill_q,self).__init__()
        self.input_layer = nn.Linear(input_dim, embed_dim_1)
        self.ffn= nn.Sequential(
            nn.Linear(embed_dim_1, embed_dim_2),
            nn.ReLU(),
            nn.Linear(embed_dim_2,embed_dim_1))
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embed_dim_1)
        self.pred_head = FCNorm(embed_dim_1, num_classes)
        self.softmax = nn.Softmax(dim=-1)
        self.cls_head = FCNorm(embed_dim_1, 2)
        
    def forward(self, inputs):
        feats = self.input_layer(inputs)
        
        for i in range(3):
            ffn_outputs = self.ffn(feats)
            ffn_outputs = self.dropout(ffn_outputs)
            ffn_outputs = self.layernorm(feats + ffn_outputs)
            feats = ffn_outputs

        pred_q = self.softmax(self.pred_head(feats))
        pred_cls = self.softmax(self.cls_head(feats))
        return pred_q, feats

def generate_seed():
    seed = os.getpid() + int(time.time() * 1e5 % 1e6)
    return seed

def main(args):
    device = args.device
    model = DQN.load("Rocket_agent_withwind/model/dqn_lunar_v0.pkl",device='cpu')

    pre_model1 = distill_q()
    pre_model1.load_state_dict(torch.load('model_final.pt'))
    
    pre_model1.eval().to(device)
    pre_model2 = Criticality_model_trans()
    pre_model2.load_state_dict(torch.load("/mnt1/brx/Rocketdata/model/stage3/dqn_model_50.pt"))
      
    pre_model2.eval().to(device)
    env = gym.make('LunarLander/ordinary_nade-v0',gravity=-8.5,enable_wind=True,wind_power = 10.0,turbulence_power = 1.0, criticality_model1 = pre_model1,criticality_model2 = pre_model2,device=device)
    
    max_epoch = 10000000
    start_epoch = args.batch_idx * 500000
    end_epoch = (args.batch_idx+1) * 500000
        
    
    return_list = []
    episode_result = []
    failure_num = 0
    
    rewards = []
    crashes = []
    crash_num = 0
    pos_num = 0
    pos_set = 0
    pos = []
    
    for epoch in tqdm.tqdm(range(start_epoch, end_epoch)):  
    #for epoch in range(1):
        #obs,_= env.reset(seed=generate_seed())
        data_episode={'obs':[],'action':[],'wind_idx':[],'torque_idx':[],'add_idx':[],'reward':[],'crash':0}
        obs,_= env.reset()
        done = False
        reward = 0
        
        episode_data = {}
        episode_data['initial_obs'] = obs.tolist()+[0,0]
        episode_data['initial_criticality'] = 1
        episode_data['initial_weight'] = 1
        
        weight_step_info = {}
        drl_epsilon_step_info = {}
        real_epsilon_step_info = {}
        criticality_step_info = {}
        ndd_step_info = {}
        q_step_info = {}
        drl_obs_step_info = {}
        
        crashes = []
        env_action = 0
        weight = 1
        control_step = 0
        i = 0
        end_time = 0
        #print('-'*10)
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward_, terminated, truncated, info = env.step(action)
            
            crash = info['crash']
            data_episode['obs'].append(obs)                  
            data_episode['action'].append(action)
            data_episode['wind_idx'].append(env.wind_idx)   
            data_episode['torque_idx'].append(env.torque_idx)
            data_episode['add_idx'].append(env.add_idx)      
            done =  terminated or truncated
            crash = info['crash']
            #crash = int(truncated)
            #print(env.wind_idx,env.torque_idx)
    
            if info['criticality_info'].keys():
                all_inputs = info['criticality_info']['all_inputs']
                criticality = info['criticality_info']['criticality']
                pdf_array = info['criticality_info']['pdf_array']
                p_list = info['criticality_info']['p_list']
                action_idx = info['criticality_info']['env_action']
                cur_weight = p_list[action_idx] / pdf_array[action_idx]
                wind_mag = info['wind_mag']
                torque_mag = info['torque_mag']
                weight *= cur_weight
    
                #inputs = all_inputs[action_idx]
                
                
                if cur_weight == 1:
                    pass
                else: 
                    control_step += 1
                    weight_step_info[f'{i}'] = cur_weight
                    drl_epsilon_step_info[f'{i}'] = 1
                    real_epsilon_step_info[f'{i}'] = 1
                    q_step_info[f'{i}'] = pdf_array[action_idx]
                    ndd_step_info[f'{i}'] = p_list[action_idx]
                    criticality_step_info[f'{i}'] = criticality[action_idx]
                    drl_obs_step_info[f'{i}'] = (obs.tolist()+[float(wind_mag), float(torque_mag)])
                    #drl_obs_step_info[f'{i}'] = obs.tolist()
            if crash:
               pos_num += 1
               data_episode['crash']=1
               pos.append(data_episode)
    
            if done:
                crashes.append(crash * weight)
                if crash:
                    crash_num += 1
                    episode_data['weight_step_info '] = weight_step_info 
                    episode_data['drl_epsilon_step_info '] = drl_epsilon_step_info 
                    episode_data['real_epsilon_step_info'] = real_epsilon_step_info 
                    episode_data['criticality_step_info'] = criticality_step_info 
                    episode_data['ndd_step_info'] = ndd_step_info 
                    episode_data['q_step_info'] = q_step_info
                    episode_data['drl_obs_step_info'] = drl_obs_step_info
                    episode_data['weight_episode'] = float(weight)
                    episode_data['episdoe_info'] = {'id': epoch, 'start_time':0, 'end_time':end_time}
                    
                    #print(episode_data)
                    print(convert_to_serializable(episode_data))
                    
                    if weight > 1:
                        #file_name = f'/mnt1/brx/Rocketdata/d2rl_data/crash_new/{epoch}.json'
                        file_name = f'crash_data/d2rl_data/crash_new_1/crash_{epoch}.json'
                    else:
                        file_name = f'crash_data/d2rl_data/big_crash_new_1/crash_{epoch}.json'
                    with open(file_name,'w') as f:
                        json.dump(convert_to_serializable(episode_data),f)
                    #print(f'save json file {file_name}!')
            i+=1
            end_time+=1
      #print('epoch:', epoch,'crash_num:', crash_num, 'weight:',weight,'all_step:',i,'control_step:',control_step)
  
    env.close()
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_idx',type=int,default=4)
    parser.add_argument('--device',type=str,default='cuda:0')
    args = parser.parse_args()
    main(args)