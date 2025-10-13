import numpy as np
from copy import deepcopy
import gymnasium as gym
import torch

from tta.niches.box2d.env import make_env, bipedhard_custom
from tta.niches.box2d.bipedal_walker_test import BipedalWalkerTest, Env_config
from criticality.criticality_model import Mlp


class DeepCopyableWrapper(gym.Wrapper):

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memodict))
        return result
    
def bipedal_walker_criticality(i, actions, get_action):
    """
    Calculate criticality by Monte Carlo Sampling.
    """
    ENV_CONFIG = Env_config(
        name="edge",
        ground_roughness=0.6,
        pit_gap=[0, 0.8],
        stump_width=[1, 2],
        stump_height=[0.01, 0.4],
        stump_float=[0, 1],
        stair_height=[],
        stair_width=[],
        stair_steps=[])

    RENDER_MODE = False
    seed = 42
    # env = make_env("BipedalWalkerCustom", seed, RENDER_MODE, ENV_CONFIG)
    env = make_env("BipedalWalkerTest", seed, RENDER_MODE,
                ENV_CONFIG, get_action)
    num_rollouts = 2
    max_iters = 1000
    # grass
    if env.terrain_state == 0:
        failure_rate = np.zeros(10)
        # 对env的每个action计算criticality
        for n, action in enumerate(range(5, 15)):
            # 采多少个样本
            failure = np.zeros(num_rollouts)
            for j in range(num_rollouts):
                env.reset(seed=seed)
                # 每个样本最多运行多少轮
                for k in range(max_iters):
                    # actions：过去的action
                    env_action = env.get_random_action() if k >= len(actions) else actions[k]
                    _, _, _, _, info = env.step(env_action)
                    if info["position"][0] > 19*14/3:
                        failure[j] = 1
                        break
                print(f"action = {action}, j = {j}, failure = {failure[j]}")
            failure_rate[n] = failure.mean()
    # 坑/凸起
    else:
        failure_rate = np.zeros(5)
        for action in range(5):
            failure = np.zeros(num_rollouts)
            for j in range(num_rollouts):
                env.reset(seed=seed)
                for k in range(max_iters):
                    env_action = env.get_random_action() if k >= len(actions) else actions[k]
                    _, _, _, _, info = env.step(env_action)
                    if info["position"][0] > 19*14/3:
                        failure[j] = 1
                        break
                print(f"action = {action}, j = {j}, failure = {failure[j]}")
            failure_rate[action] = failure.mean()
    return failure_rate

def bipedal_walker_criticality_new(i, actions, get_action, model_path, state):
    """
    Calculate criticality by Monte Carlo Sampling.
    """
    criticality_model = Mlp()
    checkpoint = torch.load(model_path)
    criticality_model.load_state_dict(checkpoint['state_dict'])
    criticality_model.eval()

    ENV_CONFIG = Env_config(
        name="edge",
        ground_roughness=0.6,
        pit_gap=[0, 0.8],
        stump_width=[1, 2],
        stump_height=[0.01, 0.4],
        stump_float=[0, 1],
        stair_height=[],
        stair_width=[],
        stair_steps=[])

    RENDER_MODE = False
    seed = 42
    # env = make_env("BipedalWalkerCustom", seed, RENDER_MODE, ENV_CONFIG)
    env = make_env("BipedalWalkerTest", seed, RENDER_MODE,
                ENV_CONFIG, get_action)

    # grass
    if env.terrain_state == 0:
        failure_rate = np.zeros(10)
        # 对env的每个action计算criticality
        for n, action in enumerate(range(5, 15)):
            # 采多少个样本
            failure = criticality_model(state)
            # print(f"action = {action}, j = {j}, failure = {failure[j]}")
            failure_rate[n] = failure
    # 坑/凸起
    else:
        failure_rate = np.zeros(5)
        for action in range(5):
            failure = criticality_model(state)
            # print(f"action = {action}, j = {j}, failure = {failure[j]}")
            failure_rate[action] = failure
    return failure_rate
