import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import gymnasium as gym
import gym_testenvs

from stable_baselines3 import DQN,PPO
from stable_baselines3.common.evaluation import evaluate_policy

import numpy as np
import matplotlib.pyplot as plt

import pickle
import argparse 
import tqdm

import torch    

def main(args):

    n=600000

    batch_idx = args.batch_idx
    device = args.model_device

    env = gym.make('LunarLander/ordinary-v0',gravity=-8.5,enable_wind=True,wind_power = 10.0,turbulence_power = 1.0)

    model = DQN.load("/home/teamcommon/tyy/MyLander/Rocket_agent_withwind/model/dqn_lunar_v0.pkl",device=device)

    pos = []
    neg = []
    pos_num = 0
    neg_num = 0 
    pos_set = 0
    neg_set = 0
    for i in tqdm.tqdm(range(n)):
        data_episode={'obs':[],'action':[],'wind_idx':[],'torque_idx':[],'add_idx':[],'reward':[],'crash':0}
        obs,_= env.reset()
        done = False
        reward = 0
        print('-'*10)
        truncated_occurred = False  # 新增：跟踪是否发生truncated
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward_, terminated, truncated, info = env.step(action)
            crash = info['crash']
            data_episode['obs'].append(obs)                 
            data_episode['action'].append(action)
            data_episode['wind_idx'].append(env.wind_idx)   
            data_episode['torque_idx'].append(env.torque_idx)
            data_episode['add_idx'].append(env.add_idx)     
            data_episode['reward'].append(reward_)
            done =  terminated or truncated
            if truncated:
                truncated_occurred = True #新增
            reward+=reward_
            #print(env.wind_idx,env.torque_idx)
        if crash or truncated_occurred:#修改，让truncated出现了也算crash
            pos_num+=1
            data_episode['crash']=1
            pos.append(data_episode)
        else:
            neg_num+=1
            neg.append(data_episode)    
        if pos_num == 20:
            pos_set+=1
            arr_pos = np.array(pos)
            #np.save('/mnt1/brx/Rocketdata/new_dataset_450/positive/pos_450_'+str(batch_idx)+'_'+str(pos_set)+'.npy',arr_pos)
            np.save('/mnt/mnt1/tyy/data/positive/pos_450_'+str(batch_idx)+'_'+str(pos_set)+'.npy',arr_pos)
            pos = []
            pos_num = 0
        
        if neg_num == 200:
            neg_set+=1
            arr_neg = np.array(neg)
            np.save('/mnt/mnt1/tyy/data/newnegative/neg_450_'+str(batch_idx)+'_'+str(neg_set)+'.npy',arr_neg)
            neg = []
            neg_num = 0
        
        print(f'pos num = {pos_num}')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_idx',type=int,default=4)
    parser.add_argument('--model_device',type=str,default='cuda:0')
    args = parser.parse_args()
    main(args)

