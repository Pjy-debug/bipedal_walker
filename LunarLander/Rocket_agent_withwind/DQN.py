import gymnasium as gym
import gym_testenvs
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

env = gym.make('LunarLander/train-v0',enable_wind=True,wind_power = 10.0,turbulence_power = 1.0)
model = DQN(policy = 'MlpPolicy',
            env=env,
            learning_rate =  6.3e-4,
            batch_size = 128,
            buffer_size = 50000,
            learning_starts = 0,
            gamma = 0.99,
            target_update_interval = 250,
            gradient_steps = -1,
            exploration_fraction = 0.12,
            exploration_final_eps = 0.1,
            policy_kwargs= {"net_arch" : [256, 256]})
logger=configure("logger/DQN",["tensorboard","stdout","csv"])
model.set_logger(logger)
model.learn(total_timesteps=4e5,log_interval=4,progress_bar=True)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
model.save("model/dqn_lunar_v1.pkl")