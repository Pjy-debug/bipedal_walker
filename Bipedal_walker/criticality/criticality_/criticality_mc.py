import numpy as np
from copy import deepcopy
import gymnasium as gym
from pjy.Bipedal_walker.criticality.not_that_important.copy_test import build_paralell_env, build_env
#import torch
from scipy.stats import norm
import math

from tta.niches.box2d.env import make_env, bipedhard_custom
from tta.niches.box2d.bipedal_walker_test import BipedalWalkerTest, Env_config


class DeepCopyableWrapper(gym.Wrapper):

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memodict))
        return result
    
def bipedal_walker_criticality(env, get_action):
    print(type(env))
    terrain = env.terrain
    hull = env.hull
    fd_polygon = env.fd_polygon
    fd_edge = env.fd_edge
    legs = env.legs
    joints = env.joints
    drawlist = env.drawlist
    lidar = env.lidar
    vel = env.vel
    observation_space = env.observation_space
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
    """
    # env = make_env("BipedalWalkerCustom", seed, RENDER_MODE, ENV_CONFIG)
    new_env = make_env("BipedalWalkerAdv", seed, RENDER_MODE,
                ENV_CONFIG, get_action)
    """
    num_rollouts = 1000
    max_iters = 160
    # grass
    if env.terrain_state == 0:
        failure_rate = np.zeros(10)
        # 对env的每个action计算criticality
        for n, action in enumerate(range(5, 15)):
            # 采多少个样本
            failure = np.zeros(num_rollouts)
            # 从当前状态往后跑几次，需要单独开几个环境
            for j in range(num_rollouts):
                
                #cur_env = deepcopy(env)
                cur_env = build_paralell_env(env, terrain, hull, fd_polygon, fd_edge, legs, joints, drawlist, lidar, vel, observation_space)
                _, _, _, _, info = cur_env.step(action)
                # env.reset(seed=seed)
                # 每个样本最多运行多少轮
                for k in range(max_iters):
                    # actions：过去的action
                    # env_action = cur_env.get_random_action() if k >= len(actions) else actions[k]
                    ENV_ACT = cur_env.env_act
                    if ENV_ACT:
                        if cur_env.terrain_state == 0:
                            env_action = np.random.randint(5,15)
                        else:
                            env_action = np.random.randint(5)
                    else:
                        env_action = None
                    _, _, _, _, info = cur_env.step(env_action)
                    if info["position"][0] > 19*14/3:
                        failure[j] = 1
                        break
                #print(f"action = {action}, j = {j}, failure = {failure[j]}")
            failure_rate[n] = failure.mean()
    # 坑/凸起
    else:
        failure_rate = np.zeros(5)
        for action in range(5):
            failure = np.zeros(num_rollouts)
            for j in range(num_rollouts):
                #env.reset(seed=seed)
                #cur_env = deepcopy(env)
                cur_env = build_paralell_env(env, terrain, hull, fd_polygon, fd_edge, legs, joints, drawlist, lidar, vel, observation_space)
                _, _, _, _, info = cur_env.step(action)
                for k in range(max_iters):
                    #env_action = cur_env.get_random_action() if k >= len(actions) else actions[k]
                    ENV_ACT = cur_env.env_act
                    if ENV_ACT:
                        if cur_env.terrain_state == 0:
                            env_action = np.random.randint(5, 15)
                        else:
                            env_action = np.random.randint(5)
                    else:
                        env_action = None
                    _, _, _, _, info = cur_env.step(env_action)
                    if info["position"][0] > 19*14/3:
                        failure[j] = 1
                        break
                #print(f"action = {action}, j = {j}, failure = {failure[j]}")
            failure_rate[action] = failure.mean()
    return failure_rate

def bipedal_walker_criticality_parallel(env, actions, get_action):
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
    new_env = make_env("BipedalWalkerAdv", seed, RENDER_MODE,
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
                cur_env = deepcopy(env)
                _, _, _, _, info = cur_env.step(action)
                # env.reset(seed=seed)
                # 每个样本最多运行多少轮
                for k in range(max_iters):
                    # actions：过去的action
                    env_action = cur_env.get_random_action() if k >= len(actions) else actions[k]
                    _, _, _, _, info = cur_env.step(env_action)
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
                #env.reset(seed=seed)
                cur_env = deepcopy(env)
                _, _, _, _, info = cur_env.step(action)
                for k in range(max_iters):
                    env_action = cur_env.get_random_action() if k >= len(actions) else actions[k]
                    _, _, _, _, info = cur_env.step(env_action)
                    if info["position"][0] > 19*14/3:
                        failure[j] = 1
                        break
                print(f"action = {action}, j = {j}, failure = {failure[j]}")
            failure_rate[action] = failure.mean()
    return failure_rate


def paralell_env_running(new_env, action, max_iters):
    _, _, _, _, info = new_env.step(action)
    # env.reset(seed=seed)
    # 每个样本最多运行多少轮
    for k in range(max_iters):
        # actions：过去的action
        env_action = cur_env.get_random_action() if k >= len(actions) else actions[k]
        _, _, _, _, info = cur_env.step(env_action)
        if info["position"][0] > 19 * 14 / 3:
            failure[j] = 1
            break

def calu_true_criticality(env_step_info, get_action):
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
    """
    # env = make_env("BipedalWalkerCustom", seed, RENDER_MODE, ENV_CONFIG)
    new_env = make_env("BipedalWalkerAdv", seed, RENDER_MODE,
                ENV_CONFIG, get_action)
    """
    num_rollouts = 3000
    max_iters = 160
    # grass
    if env_step_info['terrain_state'] == 0:
        failure_rate = np.zeros(10)
        # 对env的每个action计算criticality
        for n, action in enumerate(range(5, 15)):
            # 采多少个样本
            failure = np.zeros(num_rollouts)
            # 从当前状态往后跑几次，需要单独开几个环境
            for j in range(num_rollouts):
                
                #cur_env = deepcopy(env)
                cur_env = build_env(env_step_info)
                _, _, ternimated, truncated, info = cur_env.step(action)
                if ternimated and i<=max_iters:
                    failure[j] = 1
                    print('failure!')
                # env.reset(seed=seed)
                # 每个样本最多运行多少轮
                for k in range(max_iters):
                    # actions：过去的action
                    # env_action = cur_env.get_random_action() if k >= len(actions) else actions[k]
                    ENV_ACT = cur_env.env_act
                    if ENV_ACT:
                        if cur_env.terrain_state == 0:
                            env_action = np.random.randint(5,15)
                        else:
                            env_action = np.random.randint(5)
                    else:
                        env_action = None
                    _, _, ternimated, truncated, info = cur_env.step(env_action)
                    #print(info['cur_step_poly'])
                    if ternimated and i<=max_iters:
                        failure[j] = 1
                        print('failure!')
                      
                    """
                    if info["position"][0] > 19*14/3:
                        failure[j] = 1
                        break
                    """
                #print(f"action = {action}, j = {j}, failure = {failure[j]}")
                cur_env.close()
            failure_rate[n] = failure.mean()
            print(action, failure.mean())
    # 坑/凸起
    else:
        failure_rate = np.zeros(5)
        for action in range(5):
            failure = np.zeros(num_rollouts)
            for j in range(num_rollouts):
                #env.reset(seed=seed)
                #cur_env = deepcopy(env)
                cur_env = build_env(env_step_info)
                _, _, ternimated, truncated, info = cur_env.step(action)
                if ternimated and i<=max_iters:
                    failure[j] = 1
                    print('failure!')
                        
                for k in range(max_iters):
                    #env_action = cur_env.get_random_action() if k >= len(actions) else actions[k]
                    ENV_ACT = cur_env.env_act
                    if ENV_ACT:
                        if cur_env.terrain_state == 0:
                            env_action = np.random.randint(5, 15)
                        else:
                            env_action = np.random.randint(5)
                    else:
                        env_action = None
                    _, _, ternimated, truncated, info = cur_env.step(env_action)
                    if ternimated and i<=max_iters:
                        failure[j] = 1
                        print('failure!')
                    """
                    if info["position"][0] > 19*14/3:
                        failure[j] = 1
                        break
                    """
                #print(f"action = {action}, j = {j}, failure = {failure[j]}")
                cur_env.close()
            failure_rate[action] = failure.mean()
            print(action, failure.mean())
    return failure_rate


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