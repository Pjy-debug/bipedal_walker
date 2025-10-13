
import os
import time
import numpy as np
from tta.niches.box2d.model import Model

from tta.niches.box2d.env import make_env, bipedhard_custom
from tta.niches.box2d.bipedal_walker_custom import BipedalWalkerCustom, Env_config
from tta.niches.box2d.bipedal_walker_adv_new import BipedalWalkerAdv
import gymnasium as gym
import Box2D
import copy


log_dir = "logs/poet_new_test/"
# log_dir = "logs/poet_final_test/"
best_model_final_test = "logs/poet_final_test/poet_final_test.48542109-e29d-4fbe-8405-6618c92e990a.best.json"
best_model_new_test = "logs/poet_new_test/poet_new_test.966593ea-181f-42ec-ad1d-2ca11487bea9.best.json"
file_names = os.listdir(log_dir)
best_models = [log_dir + file_name for file_name in file_names if file_name.endswith('.json')]
best_models = sorted(best_models, reverse=True)
model = Model(bipedhard_custom)


env_config_1 = Env_config(
    name="edge",
    ground_roughness=0.6,
    pit_gap=[0, 0.8],
    stump_width=[1, 2],
    stump_height=[0.01, 0.4],
    stump_float=[0, 1],
    stair_height=[],
    stair_width=[],
    stair_steps=[])

ENV_CONFIG = env_config_1
RENDER_MODE = True
RGB_ARRAY = False
SAVE_REWARD = False
seed = 42
max_epoch = 1500000
best_model = best_models[39]
model.load_model(best_model)
# env = make_env("BipedalWalkerCustom", seed, RENDER_MODE, ENV_CONFIG)

env = make_env("BipedalWalkerAdv", seed, RENDER_MODE,
               ENV_CONFIG, model.get_action)

print('successfully build env')

class new_env(BipedalWalkerAdv):
    def __init__(self, env, terrain=None, hull=None, fd_polygon=None, fd_edge=None,legs=None,joints=None,drawlist=None,lidar=None,vel=None,observation_space=None):
        self.world = Box2D.b2World()
        self.terrain = terrain
        self.hull = hull
        self.fd_polygon = fd_polygon
        self.fd_edge = fd_edge
        self.legs = legs
        self.joints = joints
        self.drawlist = drawlist
        self.lidar = lidar
        self.vel = vel
        self.observation_space = observation_space
        for k,v in env.__dict__.items():
            #print(k)
            if k in ['world', 'terrain','hull','fd_polygon','fd_edge','legs','joints','drawlist','lidar','vel','observation_space']:
                continue
            else:
                self.__dict__[k] = copy.deepcopy(v)

def build_paralell_env(env, terrain=None, hull=None, fd_polygon=None, fd_edge=None,legs=None,joints=None,drawlist=None,lidar=None,vel=None,observation_space=None):
    new_terrain = terrain
    new_hull = hull
    new_fd_polygon = fd_polygon
    new_fd_edge = fd_edge
    new_legs = legs
    new_joints = joints
    new_drawlist = drawlist
    new_lidar = lidar
    new_vel = vel
    new_observation_space = observation_space
    paralell_env = new_env(env, new_terrain, new_hull, new_fd_polygon, new_fd_edge, new_legs, new_joints, new_drawlist, new_lidar, new_vel, new_observation_space)
    return paralell_env

def save_collision_step(env, terrain, hull, fd_polygon, fd_edge, legs, joints, drawlist, lidar,vel,observation_space):
    env_info = {}
    env_info['terrain'] = terrain
    env_info['hull'] = hull
    env_info['fd_polygon'] = fd_polygon
    env_info['fd_edge'] = fd_edge
    env_info['legs'] = legs
    env_info['joints'] = joints
    env_info['drawlist'] = drawlist
    env_info['lidar'] = lidar
    env_info['vel'] = vel
    env_info['observation_space'] = observation_space
    for k,v in env.__dict__.items():
        #print(k)
        if k in ['world', 'terrain','hull','fd_polygon','fd_edge','legs','joints','drawlist','lidar','vel','observation_space']:
            continue
        else:
            env_info[k] = copy.deepcopy(v)
    return env_info

class new_env_(BipedalWalkerAdv):
    def __init__(self, env_step_info):
        self.world = Box2D.b2World()
        for k,v in env_step_info.items():
            self.__dict__[k] = v
            
        """
        self.terrain = env_step_info['terrain']
        self.hull = env_step_info['hull']
        self.fd_polygon = env_step_info['fd_polygon']
        self.fd_edge = env_step_info['fd_edge']
        self.legs = env_step_info['legs']
        self.joints = env_step_info['joints']
        self.drawlist = env_step_info['drawlist']
        self.lidar = env_step_info['lidar']
        self.vel = env_step_info['vel']
        self.observation_space = env_step_info['observation_space']
        for k,v in env_step_info.items():
            #print(k)
            if k in ['world', 'terrain','hull','fd_polygon','fd_edge','legs','joints','drawlist','lidar','vel','observation_space']:
                continue
            else:
                self.__dict__[k] = v
        """
def build_env(env_step_info):
    paralell_env = new_env_(env_step_info)
    return paralell_env

    












    