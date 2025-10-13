import gymnasium as gym
import gym_testenvs
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('LunarLander/ordinary-v0',enable_wind=True,wind_power = 10.0,turbulence_power = 1.0,render_mode='human')
model = DQN.load("model/dqn_lunar_v0.pkl")

def mean_value_test():
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000,render=False)#render=True
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

mean_value_test()