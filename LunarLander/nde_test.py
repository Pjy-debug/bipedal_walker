import gymnasium as gym
import gym_testenvs

from stable_baselines3 import DQN,PPO
from stable_baselines3.common.evaluation import evaluate_policy

import numpy as np
import matplotlib.pyplot as plt

import pickle
import tqdm
import math
import random
from scipy.stats import norm

def plot_mean(ys,xlabel,ylabel,legend,xlim,save_path=None):
    plt.plot(ys)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.xlim(xlim)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

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

n=500000

#env = gym.make('LunarLander/ordinary-v0',gravity=-8.5,enable_wind=True,wind_power = 10.0,turbulence_power = 1.0)
env = gym.make('LunarLander/nde-v0',gravity=-8.5,enable_wind=True,wind_power = 10.0,turbulence_power = 1.0)

model = DQN.load("Rocket_agent_withwind/model/dqn_lunar_v0.pkl",device='cpu')

rewards = []
crashes = []
failure_num = 0
for i in range(n):
    obs,_= env.reset()
    done = False
    reward = 0
    step_num = 0
    while not done:
        step_num += 1
        action, _states = model.predict(obs, deterministic=True)
        obs, reward_, terminated, truncated, info = env.step(action)
        done =  terminated or truncated
        crash = info['crash']
        #crash = int(truncated)
        reward+=reward_
        if done:
            
            rewards.append(reward)
            crashes.append(crash)
            if crash:
                failure_num += 1
    print('epoch:',i,step_num,'failure:',crash)
    if i % 5 == 0:
        Mean, RHF, Val = calculate_val(crashes)
        print('RHF:',RHF[-1],'val:',Val[-1],'Mean:',sum(crashes)/len(crashes),'failure_num:',failure_num)
    """
    if i % 10000 == 0:
        np.save(f'data/crash_new_450_orin_{i}.npy', crashes)
    """
#np.save('data/crashes_mean_new_450_orin.npy',crashes)
#print(crashes)
dict={'rewards':rewards,'crashes':crashes}
with open('data/ordinary/DQN/ordinary.pkl','wb') as f:
    pickle.dump(dict,f)
    f.close()

mean_reward,rhw_reward,var_reward= calculate_val(rewards)
mean_crash,rhw_crash,var_crash= calculate_val(crashes)

print(mean_crash)

plot_mean(mean_reward,'Number of Episodes','Reward',['Ordinary method'],[0,n],save_path='data/ordinary/DQN/reward/ordinary_mean.png')
plot_mean(rhw_reward,'Number of Episodes','Relative Half Width',['Ordinary method'],[0,n],save_path='data/ordinary/DQN/reward/ordinary_rhw.png')
plot_mean(var_reward,'Number of Episodes','Variance',['Ordinary method'],[0,n],save_path='data/ordinary/DQN/reward/ordinary_var.png')

plot_mean(mean_crash,'Number of Episodes','Crash Rate',['Ordinary method'],[0,n],save_path='data/ordinary/DQN/crash/ordinary_mean.png')
plot_mean(rhw_crash,'Number of Episodes','Relative Half Width',['Ordinary method'],[0,n],save_path='data/ordinary/DQN/crash/ordinary_rhw.png')
plot_mean(var_crash,'Number of Episodes','Variance',['Ordinary method'],[0,n],save_path='data/ordinary/DQN/crash/ordinary_var.png')



    