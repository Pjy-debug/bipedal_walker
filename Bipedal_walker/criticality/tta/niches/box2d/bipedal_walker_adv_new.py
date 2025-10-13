# The following code is modified from openai/gym (https://github.com/openai/gym) under the MIT License.

# Modifications Copyright (c) 2020 Uber Technologies, Inc.
'''*************************************************************************
[file name]                 bipedal_walker_adv_new.py
[description]              BipedalWalkerAdv类文件
[developer]                brx, 2024
[changelog]               （若修改过则必需注明）
[usage]                    
*************************************************************************'''

import sys
import math
import numpy as np
from typing import Optional

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef,
                      polygonShape, revoluteJointDef, contactListener)

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import colorize, seeding
from collections import namedtuple

# This is simple 4-joints walker robot environment.
#
# There are two versions:
#
# - Normal, with slightly uneven terrain.
#
# - Hardcore with ladders, stumps, pitfalls.
#
# Reward is given for moving forward, total 300+ points up to the far end. If the robot falls,
# it gets -100. Applying motor torque costs a small amount of points, more optimal agent
# will get better score.
#
# Heuristic is provided for testing, it's also useful to get demonstrations to
# learn from. To run heuristic:
#
# python gym/envs/box2d/bipedal_walker.py
#
# State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
# position of joints and joints angular speed, legs contact with ground, and 10 lidar
# rangefinder measurements to help to deal with the hardcore version. There's no coordinates
# in the state vector. Lidar is less useful in normal version, but it works.
#
# To solve the game you need to get 300 points in 1600 time steps.
#
# To solve hardcore version you need 300 points in 2000 time steps.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

Env_config = namedtuple('Env_config', [
    'name',
    'ground_roughness',
    'pit_gap',
    'stump_width',  'stump_height', 'stump_float',
    'stair_height', 'stair_width', 'stair_steps'
])

FPS = 50
SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 80
SPEED_HIP = 4
SPEED_KNEE = 6
LIDAR_RANGE = 160 / SCALE

INITIAL_RANDOM = 5

HULL_POLY = [
    (-30, +9), (+6, +9), (+34, +1),
    (+34, -8), (-30, -8)
]
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

BOTTOM = 0 # viewer lower boundary, divided by SCALE

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 20    # in steps
FRICTION = 2.5

HULL_FD = fixtureDef(
    shape=polygonShape(vertices=[(x / SCALE, y / SCALE)
                                 for x, y in HULL_POLY]),
    density=5.0,
    friction=0.1,
    categoryBits=0x0020,
    maskBits=0x001,  # collide only with ground
    restitution=0.0)  # 0.99 bouncy

LEG_FD = fixtureDef(
    shape=polygonShape(box=(LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001)

LOWER_FD = fixtureDef(
    shape=polygonShape(box=(0.8 * LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001)


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.hull == contact.fixtureA.body or self.env.hull == contact.fixtureB.body:
            self.env.game_over = True
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = True

    def EndContact(self, contact):
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False


class BipedalWalkerAdv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __repr__(self):
        return "{}\nenv\n{}".format(self.__dict__, self.__dict__["np_random"].get_state())

    def __init__(
            self,
            env_config,
            get_agent_action, # function
            seed = 42
        ):
        self.seed_ = seed
        self.spec = None
        self.set_env_config(env_config)
        self.env_params = None
        self.env_seed = None
        self._seed(self.seed_)
        self.viewer = None
        self.get_agent_action = get_agent_action
        self.state = None
        self.env_act = False

        self.world = Box2D.b2World()
        self.terrain = None
        self.hull = None

        self.terrain_by_step = []
        self.env_actions = []
        self.all_pos = []

        self.prev_shaping = None
        self.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=[(0, 0),
                                         (1, 0),
                                         (1, -1),
                                         (0, -1)]),
            friction=FRICTION)

        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=[(0, 0),
                                      (1, 1)]),
            friction=FRICTION,
            categoryBits=0x0001,
        )

        self._reset(self.seed_)

        high = np.array([np.inf] * 24)
        self.action_space = spaces.Box(
            np.array([-1, -1, -1, -1]), np.array([+1, +1, +1, +1]))
        self.observation_space = spaces.Box(-high, high)

    def set_env_config(self, env_config):
        self.config = env_config

    def augment(self, params):
        self.env_params = params
        print("env augmented!")

    def _set_terrain_number(self):
        self.hardcore = False
        self.GRASS = 0
        self.STUMP, self.STAIRS, self.PIT = -1, -1, -1
        self._STATES_ = 1

        # 凸起: _STATES_=2
        if self.config.stump_width and self.config.stump_height and self.config.stump_float:
            # STUMP exist
            self.STUMP = self._STATES_
            self._STATES_ += 1

        # 台阶: _STATES_=3
        if self.config.stair_height and self.config.stair_width and self.config.stair_steps:
            # STAIRS exist
            self.STAIRS = self._STATES_
            self._STATES_ += 1

        # 坑: _STATES_=4
        if self.config.pit_gap:
            # PIT exist
            self.PIT = self._STATES_
            self._STATES_ += 1

        # 有坑或障碍物就是难
        if self._STATES_ > 1:
            self.hardcore = True

    def save_env_def(self, filename):
        import json
        a = {'config': self.config._asdict(), 'seed': self.env_seed}
        with open(filename, 'w') as f:
            json.dump(a, f)

    def seed(self, seed=None):
        return self._seed(seed)

    def _seed(self, seed=None):
        self.env_seed = seed
        self.np_random, seed = seeding.np_random(seed)
        # print(f"seed = {seed}")
        return [seed]

    def _destroy(self):
        if not self.terrain:
            return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        self.world.DestroyBody(self.hull)
        self.hull = None
        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []
        self.joints = []
        self.world = None

    def _generate_terrain(self, hardcore, debug=False):
        # GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state = self.GRASS
        velocity = 0.0
        y = TERRAIN_HEIGHT
        counter = TERRAIN_STARTPAD
        oneshot = False
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []
        pit_diff = 0
        for i in range(TERRAIN_LENGTH):
            x = i * TERRAIN_STEP
            self.terrain_x.append(x)
            if debug: print(f"i = {i}, x = {x}")

            if state == self.GRASS and not oneshot:
                velocity = 0.8 * velocity + 0.01 * np.sign(TERRAIN_HEIGHT - y)
                if self.env_params is not None and self.env_params.altitude_fn is not None:
                    y += velocity
                    if i > TERRAIN_STARTPAD:
                        mid = TERRAIN_LENGTH * TERRAIN_STEP / 2.
                        x_ = (x - mid) * np.pi / mid
                        y = TERRAIN_HEIGHT + self.env_params.altitude_fn((x_, ))[0]
                        if i == TERRAIN_STARTPAD+1:
                            y_norm = self.env_params.altitude_fn((x_, ))[0]
                        y -= y_norm
                    if debug: print(f"cppn used, i = {i}, y = {y}")
                else:
                    if i > TERRAIN_STARTPAD:
                        velocity += self.np_random.uniform(-1, 1) / SCALE  # 1
                    # input parameter: ground_roughness
                    #ground_roughness = 1
                    y += self.config.ground_roughness * velocity
                    if debug: print(f"y = {y}")

            elif state == self.PIT and oneshot:
                # input parameter: pit_gap
                # pit_gap = self.np_random.integers(3, 5) #counter is the control of the GAP distance
                #counter = pit_gap
                #counter = self.np_random.integers(*self.config.pit_gap)
                pit_gap = 1.0 + self.np_random.uniform(*self.config.pit_gap)
                if debug: print(f"pit_gap = {pit_gap}")
                counter = np.ceil(pit_gap)
                pit_diff = counter - pit_gap

                poly = [
                    (x,              y),
                    (x + TERRAIN_STEP, y),
                    (x + TERRAIN_STEP, y - 4 * TERRAIN_STEP),
                    (x,              y - 4 * TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(
                    fixtures=self.fd_polygon)
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices = [
                    (p[0] + TERRAIN_STEP * pit_gap, p[1]) for p in poly]
                t = self.world.CreateStaticBody(
                    fixtures=self.fd_polygon)
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state == self.PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4 * TERRAIN_STEP
                if counter == 1:
                    self.terrain_x[-1] = self.terrain_x[-1] - pit_diff * TERRAIN_STEP
                    pit_diff = 0

            elif state == self.STUMP and oneshot:
                # input parameter stump_width, stump_height, stump_float
                #stump_width = self.np_random.uniform(*self.config.stump_width)
                stump_width = self.np_random.integers(*self.config.stump_width)
                stump_height = self.np_random.uniform(
                    *self.config.stump_height)
                stump_float = self.np_random.integers(*self.config.stump_float)
                #counter = np.ceil(stump_width)
                if debug: print(f"stump_width = {stump_width}, stump_height = {stump_height}, stump_float = {stump_float}")
                counter = stump_width
                countery = stump_height
                poly = [
                    (x,                      y + stump_float * TERRAIN_STEP),
                    (x + stump_width * TERRAIN_STEP, y + stump_float * TERRAIN_STEP),
                    (x + stump_width * TERRAIN_STEP, y + countery *
                     TERRAIN_STEP + stump_float * TERRAIN_STEP),
                    (x,                      y + countery *
                     TERRAIN_STEP + stump_float * TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(
                    fixtures=self.fd_polygon)
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)

            elif state == self.STAIRS and oneshot:
                # input parameters: stair_height, stair_width, stair_steps
                stair_height = self.np_random.uniform(
                    *self.config.stair_height)
                stair_slope = 1 if self.np_random.rand() > 0.5 else -1
                stair_width = self.np_random.integers(*self.config.stair_width)
                stair_steps = self.np_random.integers(*self.config.stair_steps)
                if debug: print(f"stair_slope = {stair_slope}, stair_width = {stair_width}, stair_steps = {stair_steps}")
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (x + (s * stair_width) * TERRAIN_STEP, y +
                         (s * stair_height * stair_slope) * TERRAIN_STEP),
                        (x + ((1 + s) * stair_width) * TERRAIN_STEP, y +
                         (s * stair_height * stair_slope) * TERRAIN_STEP),
                        (x + ((1 + s) * stair_width) * TERRAIN_STEP, y +
                         (-stair_height + s * stair_height * stair_slope) * TERRAIN_STEP),
                        (x + (s * stair_width) * TERRAIN_STEP, y + (-stair_height +
                                                                    s * stair_height * stair_slope) * TERRAIN_STEP),
                    ]
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(
                        fixtures=self.fd_polygon)
                    t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                    self.terrain.append(t)
                counter = stair_steps * stair_width + 1

            elif state == self.STAIRS and not oneshot:
                s = stair_steps * stair_width - counter
                n = s // stair_width
                y = original_y + (n * stair_height * stair_slope) * TERRAIN_STEP - \
                    (stair_height if stair_slope == -1 else 0) * TERRAIN_STEP

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter == 0:
                counter = self.np_random.integers(
                    TERRAIN_GRASS / 2, TERRAIN_GRASS)
                if state == self.GRASS and hardcore:
                    state = self.np_random.integers(1, self._STATES_)
                    oneshot = True
                else:
                    state = self.GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH - 1):
            poly = [
                (self.terrain_x[i],   self.terrain_y[i]),
                (self.terrain_x[i + 1], self.terrain_y[i + 1])
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(
                fixtures=self.fd_edge)
            color = (0.3, 1.0 if i % 2 == 0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [(poly[1][0], BOTTOM), (poly[0][0], BOTTOM)]
            # poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))
        self.terrain.reverse()

    def _init_env(self, debug=False):
        # 初始生成grass
        self.terrain_state = self.GRASS
        self.terrain_velocity = 0.0
        # 初始草地高度
        self.current_y = TERRAIN_HEIGHT
        # 生成了多少个
        self.terrain_counter = 0
        self.terrain_oneshot = False
        # 生成的地形
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []
        self.env_actions = []
        self.all_pos = []
        self.pit_diff = 0

        self.weights = []
        for i in range(TERRAIN_STARTPAD):
            x = i * TERRAIN_STEP
            # 每一段草地开始的位置
            self.terrain_x.append(x)
            self.terrain_y.append(self.current_y)
            self.env_actions.append({'current_i':i,'start_x':x,'pred_i':None,'terrain_counter':self.terrain_counter,
                                     'terrain_type': self.terrain_state,'states': self._STATES_,'action':None,
                                     'current_y':self.current_y})
            if debug: print(f"i = {i}, x = {round(x, 2)}, y = {round(self.current_y, 2)}")

        self.current_i = i+1

        self.terrain_poly = []
        for i in range(TERRAIN_STARTPAD - 1):
            poly = [
                (self.terrain_x[i],   self.terrain_y[i]),
                (self.terrain_x[i + 1], self.terrain_y[i + 1])
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(
                fixtures=self.fd_edge)
            color = (0.3, 1.0 if i % 2 == 0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [(poly[1][0], BOTTOM), (poly[0][0], BOTTOM)]
            # poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))
        self.terrain.reverse()


    def _step_env(self, action, q, hardcore=True, sparse_IS=False, debug=False):
        #print('env step!')
        #print('current_i=',self.current_i,'pred_i=',self.pred_i,'agent_i=',self.agent_i)
        # action : env_action
        self.terrain.reverse()
        env_actions = []
        init_i = self.current_i
        cur_step_poly = []
        input_action = action

        # if debug: print(f"init_i = {init_i}, pred_i = {pred_i}, pred_bound = {pred_bound}")

        #print('-------env take actions!---------')
        while self.current_i < self.pred_i or self.terrain_counter > 0:

            env_action = {}
            env_action['current_i'] = self.current_i
            env_action['pred_i'] = self.pred_i
            # self.terrain_cls.append(self.terrain_state)

            # 本次更新的开始x值
            x = self.current_i * TERRAIN_STEP
            self.terrain_x.append(x)
            env_action['start_x'] = x
            env_action['input_action'] = action

            self.current_i += 1
            if debug: print(f"i = {self.current_i}, x = {round(x, 2)}")
            # print('input_action:',action,'terrain_counter:',self.terrain_counter)

            if self.terrain_counter == 0:
                # print(f"self.terrain_state = {self.terrain_state}")
                # 当前生成草地
                if self.terrain_state == self.GRASS and hardcore:
                    # print(f"action = {action}")
                    # new
                    if action <= 4:
                        if input_action < 10:
                            action += 5  
                        elif input_action >= 10:
                            action += 10
                    assert action > 4 and action < 15
                    if sparse_IS:
                        self.terrain_state = 1 if action > 9 else 2
                        p = 1 / (self._STATES_ - 1)
                        #q = p
                        #print('terrain counter == 0 and terrain_state == grass, p=',p)
                        if q == None:
                            self.weights.append(1)
                        else:
                            self.weights.append(0.1 / q)
                    else:
                        # 下一个生成地形类别
                        self.terrain_state = self.np_random.integers(1, self._STATES_)
                        self.weights.append(1)
                    if debug: print(f"self.terrain_state = {self.terrain_state}")
                else:
                    # 坑、凸起后一定是草地
                    # print(f"action = {action}")
                    if action > 4 and action < 10:
                        action -= 5  # [0-4]
                    elif action > 9:
                        action -= 10
                    self.terrain_state = self.GRASS
                    if sparse_IS:
                        self.terrain_counter = action + 5
                        p = 2 / TERRAIN_GRASS   # 10
                        #q = p
                        #print('terrain counter == 0 and terrain_state != grass, terrain_counter',self.terrain_counter)
                        if q == None:
                            self.weights.append(1)
                        else:
                            self.weights.append(p / q)
                    else:
                        self.terrain_counter = self.np_random.integers(TERRAIN_GRASS / 2, TERRAIN_GRASS)
                        self.weights.append(1)
                    if debug: print(f"self.terrain_counter = {self.terrain_counter}")
                self.terrain_oneshot = True

            env_action['terrain_type'] = self.terrain_state
            env_action['states'] = self._STATES_

            if self.terrain_state == self.GRASS and not self.terrain_oneshot:
                self.terrain_velocity = 0.8 * self.terrain_velocity + \
                     0.01 * np.sign(TERRAIN_HEIGHT - self.current_y)
                if self.env_params is not None and self.env_params.altitude_fn is not None:
                    self.current_y += self.terrain_velocity
                    if i > TERRAIN_STARTPAD:
                        mid = TERRAIN_LENGTH * TERRAIN_STEP / 2.
                        x_ = (x - mid) * np.pi / mid
                        self.current_y = TERRAIN_HEIGHT + self.env_params.altitude_fn((x_, ))[0]
                        if i == TERRAIN_STARTPAD+1:
                            y_norm = self.env_params.altitude_fn((x_, ))[0]
                        self.current_y -= y_norm
                    if debug: print(f"cppn used, i = {self.current_i}, y = {self.current_y}")
                else:
                    if sparse_IS:
                        """
                        delta_v = self.np_random.uniform(-1, 1) / SCALE
                        self.terrain_velocity += delta_v
                        """
                        if action >=0 and action < 5:
                            delta_v = (action - 2 ) / 2
                        elif action >=5 and action < 15:
                            delta_v = (action - 9.5) / 4.5
                        self.terrain_velocity += delta_v  / SCALE
                        #p = 1/2
                        #q = p
                        #print('terrain counter > 0 and terrain_state == grass, delta_v',delta_v)
                        self.weights.append(1)
                    else:
                        delta_v = self.np_random.uniform(-1, 1) / SCALE
                        self.terrain_velocity += delta_v
                        self.weights.append(1)
                    env_action['action'] = delta_v
                    self.current_y += self.config.ground_roughness * self.terrain_velocity
                    if debug: print(f"y = {round(self.current_y, 2)}")

            elif self.terrain_state == self.PIT and self.terrain_oneshot:
                if sparse_IS:
                    pit_gap = 1.0 + (action - 5) * 0.2
                    #p = 1 / (self.config.pit_gap[1] - self.config.pit_gap[0])
                    #q = p
                    #print('terrain counter == 0 and terrain_state == git, pit_gap=', pit_gap)
                    self.weights.append(1)
                else:
                    pit_gap = 1.0 + self.np_random.uniform(*self.config.pit_gap)
                    self.weights.append(1)
                env_action['action'] = pit_gap
                if debug: print(f"pit_gap = {pit_gap}")

                self.terrain_counter = np.ceil(pit_gap)
                self.pit_diff = self.terrain_counter - pit_gap

                poly = [
                    (x,              self.current_y),
                    (x + TERRAIN_STEP, self.current_y),
                    (x + TERRAIN_STEP, self.current_y - 4 * TERRAIN_STEP),
                    (x,              self.current_y - 4 * TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(
                    fixtures=self.fd_polygon)
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices = [
                    (po[0] + TERRAIN_STEP * pit_gap, po[1]) for po in poly]
                t = self.world.CreateStaticBody(
                    fixtures=self.fd_polygon)
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)
                self.terrain_counter += 2
                env_action['env_poly'] = [(po[0] + TERRAIN_STEP * pit_gap, po[1]) for po in poly]
                cur_step_poly = env_action['env_poly']
                original_y = self.current_y

            elif self.terrain_state == self.PIT and not self.terrain_oneshot:
                self.current_y = original_y
                if self.terrain_counter > 1:
                    self.current_y -= 4 * TERRAIN_STEP
                if self.terrain_counter == 1:
                    self.terrain_x[-1] = self.terrain_x[-1] - self.pit_diff * TERRAIN_STEP
                    self.pit_diff = 0

            elif self.terrain_state == self.STUMP and self.terrain_oneshot:
                if sparse_IS:
                    stump_width = self.np_random.integers(*self.config.stump_width)
                    stump_height = (action - 9) * 0.1
                    if stump_height < 0:
                        stump_height = -stump_height
                    stump_float = self.np_random.integers(*self.config.stump_float)
                    #p = 1
                    #q = p
                    #print('terrain counter == 0 and terrain_state == stump', 'height=',stump_height,'width=',stump_width,'float=',stump_float)
                    self.weights.append(1)
                else:
                    stump_width = self.np_random.integers(*self.config.stump_width)
                    stump_height = self.np_random.uniform(*self.config.stump_height)
                    stump_float = self.np_random.integers(*self.config.stump_float)
                    self.weights.append(1)
                env_action['action'] = (stump_height,stump_width,stump_float)
                if debug:
                    print(f"stump_width = {stump_width}, stump_height = {stump_height}, stump_float = {stump_float}")
                self.terrain_counter = stump_width
                countery = stump_height
                poly = [
                    (x,                      self.current_y + stump_float * TERRAIN_STEP),
                    (x + stump_width * TERRAIN_STEP, self.current_y + stump_float * TERRAIN_STEP),
                    (x + stump_width * TERRAIN_STEP, self.current_y + countery *
                     TERRAIN_STEP + stump_float * TERRAIN_STEP),
                    (x,                      self.current_y + countery *
                     TERRAIN_STEP + stump_float * TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(
                    fixtures=self.fd_polygon)
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)
                env_action['env_poly'] = poly
                cur_step_poly = poly

            elif self.terrain_state == self.STAIRS and self.terrain_oneshot:
                # input parameters: stair_height, stair_width, stair_steps
                stair_height = self.np_random.uniform(
                    *self.config.stair_height)
                stair_slope = 1 if self.np_random.rand() > 0.5 else -1
                stair_width = self.np_random.integers(*self.config.stair_width)
                stair_steps = self.np_random.integers(*self.config.stair_steps)
                if debug:
                    print(f"stair_slope = {stair_slope}, stair_width = {stair_width}, stair_steps = {stair_steps}")
                original_y =  self.current_y
                for s in range(stair_steps):
                    poly = [
                        (x + (s * stair_width) * TERRAIN_STEP, self.current_y +
                         (s * stair_height * stair_slope) * TERRAIN_STEP),
                        (x + ((1 + s) * stair_width) * TERRAIN_STEP, self.current_y +
                         (s * stair_height * stair_slope) * TERRAIN_STEP),
                        (x + ((1 + s) * stair_width) * TERRAIN_STEP, self.current_y +
                         (-stair_height + s * stair_height * stair_slope) * TERRAIN_STEP),
                        (x + (s * stair_width) * TERRAIN_STEP, self.current_y + (-stair_height +
                                                                    s * stair_height * stair_slope) * TERRAIN_STEP),
                    ]
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(
                        fixtures=self.fd_polygon)
                    t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                    self.terrain.append(t)
                counter = stair_steps * stair_width + 1

            elif self.terrain_state == self.STAIRS and not self.terrain_oneshot:
                s = stair_steps * stair_width - counter
                n = s // stair_width
                self.current_y = original_y + (n * stair_height * stair_slope) * TERRAIN_STEP - \
                    (stair_height if stair_slope == -1 else 0) * TERRAIN_STEP

            self.terrain_oneshot = False
            self.terrain_y.append(self.current_y)
            self.terrain_counter -= 1
            env_action['current_y'] = self.current_y
            env_action['terrain_counter'] = self.terrain_counter

            env_actions.append(env_action)

        # current_i = pred_i
        terrain_tmp = []
        for i in range(init_i-1, self.current_i-1):
            poly = [
                (self.terrain_x[i],   self.terrain_y[i]),
                (self.terrain_x[i + 1], self.terrain_y[i + 1])
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(
                fixtures=self.fd_edge)
            color = (0.3, 1.0 if i % 2 == 0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            terrain_tmp.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [(poly[1][0], BOTTOM), (poly[0][0], BOTTOM)]
            # poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))
        self.terrain += terrain_tmp
        self.terrain.reverse()
        self.drawlist = self.terrain + self.legs + [self.hull]
        # print('start_i:',init_i,'end_i:',self.current_i,'total_step:',self.current_i-init_i)
        if not cur_step_poly:
            cur_step_poly = [(init_i*TERRAIN_STEP, self.current_y),(self.current_i * TERRAIN_STEP, self.current_y),
                             (init_i*TERRAIN_STEP, self.current_y),(self.current_i * TERRAIN_STEP, self.current_y)]
        # print('cur_step_poly',cur_step_poly)
        return env_actions, self.terrain_poly[init_i-1:self.current_i-1], init_i, self.current_i, cur_step_poly

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly = []
        for i in range(TERRAIN_LENGTH // 20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH) * TERRAIN_STEP
            y = VIEWPORT_H / SCALE * 3 / 4
            poly = [
                (x + 15 * TERRAIN_STEP * math.sin(3.14 * 2 * a / 5) + self.np_random.uniform(0, 5 * TERRAIN_STEP),
                 y + 5 * TERRAIN_STEP * math.cos(3.14 * 2 * a / 5) + self.np_random.uniform(0, 5 * TERRAIN_STEP))
                for a in range(5)]
            x1 = min([p[0] for p in poly])
            x2 = max([p[0] for p in poly])
            self.cloud_poly.append((poly, x1, x2))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        return self._reset(seed=seed)

    def get_random_action(self):
        if self.terrain_state == 0:
            action = np.random.randint(5, 15)
        else:
            action = np.random.randint(5)
        return action

    def _reset(self, seed=34):
        self._seed(seed)
        self._destroy()
        self.world = Box2D.b2World()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        self._set_terrain_number()
        # self._generate_terrain(self.hardcore)
        self._init_env()
        self._generate_clouds()

        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            fixtures=HULL_FD
        )
        self.hull.color1 = (0.5, 0.4, 0.9)
        self.hull.color2 = (0.3, 0.3, 0.5)
        self.hull.ApplyForceToCenter(
            (self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True)

        self.legs = []
        self.joints = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LEG_FD
            )
            leg.color1 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.)
            leg.color2 = (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=leg,
                localAnchorA=(0, LEG_DOWN),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=i,
                lowerAngle=-0.8,
                upperAngle=1.1,
            )
            self.legs.append(leg)
            self.joints.append(self.world.CreateJoint(rjd))

            lower = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H * 3 / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LOWER_FD
            )
            lower.color1 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.)
            lower.color2 = (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                localAnchorA=(0, -LEG_H / 2),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=-1.6,
                upperAngle=-0.1,
            )
            lower.ground_contact = False
            self.legs.append(lower)
            self.joints.append(self.world.CreateJoint(rjd))

        self.drawlist = self.terrain + self.legs + [self.hull]

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return 1
                self.p2 = point
                self.fraction = fraction
                return 0
        self.lidar = [LidarCallback() for _ in range(10)]
        action = self.get_random_action()
        return self._step(action, agent_action=np.array([0, 0, 0, 0]))[0]

    def step(self, env_action,q=None):
        return self._step(env_action,q=None)

    def _step(self, env_action, q=None, agent_action=None):
        if agent_action is not None:
            action = agent_action
        else:
            action = self.get_agent_action(self.state)
        # self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
        control_speed = False  # Should be easier as well
        if control_speed:
            self.joints[0].motorSpeed = float(
                SPEED_HIP * np.clip(action[0], -1, 1))
            self.joints[1].motorSpeed = float(
                SPEED_KNEE * np.clip(action[1], -1, 1))
            self.joints[2].motorSpeed = float(
                SPEED_HIP * np.clip(action[2], -1, 1))
            self.joints[3].motorSpeed = float(
                SPEED_KNEE * np.clip(action[3], -1, 1))
        else:
            self.joints[0].motorSpeed = float(SPEED_HIP * np.sign(action[0]))
            self.joints[0].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1))
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.sign(action[1]))
            self.joints[1].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1))
            self.joints[2].motorSpeed = float(SPEED_HIP * np.sign(action[2]))
            self.joints[2].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1))
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.sign(action[3]))
            self.joints[3].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1))

        # agent运动
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.hull.position
        vel = self.hull.linearVelocity
        self.pos = (pos[0],pos[1])
        self.vel = vel

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
                pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE)
            self.world.RayCast(
                self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        pred_bound = pos[0] / SCALE + self.lidar[-1].p2[0] + vel.x / FPS / SCALE
        self.pred_i = np.ceil(pred_bound / TERRAIN_STEP)
        self.agent_i = np.ceil(pos[0] / TERRAIN_STEP)
        #print("lidar range:",pred_bound-pos[0],self.pred_i-self.agent_i)

        if self.current_i < self.pred_i:
            self.env_act = True
            self.init_i = 0
        else:
            self.env_act = False

        if env_action is not None:
            self.new_env_actions, self.new_terrain_poly, self.cur_start, self.cur_end, self.cur_step_poly = self._step_env(env_action,q ,sparse_IS=True)
            self.env_actions += self.new_env_actions
        
        state = [
            # Normal angles up to 0.5 here, but sure more is possible.
            self.hull.angle,
            2.0 * self.hull.angularVelocity / FPS,
            # Normalized to get -1..1 range
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
            # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            self.joints[0].angle,
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0
        ]
        state += [l.fraction for l in self.lidar]
        assert len(state) == 24
        self.state = state
        #print('state:',self.state)

        self.scroll = pos.x - VIEWPORT_W / SCALE / 5

        # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping = 130 * pos[0] / SCALE
        # keep head straight, other than that and falling, any behavior is unpunished
        shaping -= 5.0 * abs(state[0])

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        terminated = False
        finish = False
        if self.game_over or pos[0] < 0:
            reward = -100
            terminated = True
        if pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            terminated = True
            finish = True
        # print(self.pos,self.agent_i,self.pred_i)
        self.all_pos.append((self.pos,self.agent_i,self.pred_i))
        info = {"position": (pos[0],pos[1]),
                'agent_i': self.agent_i,
                'lidar_pred': self.pred_i,
                'cur_step_poly': self.cur_step_poly,
                "weight": np.prod(self.weights),
                'cur_step_env_start': self.cur_start,
                'cur_step_env_end': self.cur_end,
                'next_env_actions': self.new_env_actions,
                'next_terrain_poly': self.new_terrain_poly,
                }
        return np.array(state), reward, terminated, not finish, info

    def render(self, *args, **kwargs):
        return self._render(*args, **kwargs)

    def _render(self, mode='human', close=False):
        # VIEWPORT_W = 1700
        # VIEWPORT_H = 800
        # TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(self.scroll, VIEWPORT_W /
                               SCALE + self.scroll, BOTTOM, VIEWPORT_H / SCALE)

        self.viewer.draw_polygon([
            (self.scroll,                  0),
            (self.scroll + VIEWPORT_W / SCALE, 0),
            (self.scroll + VIEWPORT_W / SCALE, VIEWPORT_H / SCALE),
            (self.scroll,                  VIEWPORT_H / SCALE),
        ], color=(0.9, 0.9, 1.0))
        for poly, x1, x2 in self.cloud_poly:
            if x2 < self.scroll / 2:
                continue
            if x1 > self.scroll / 2 + VIEWPORT_W / SCALE:
                continue
            self.viewer.draw_polygon(
                [(p[0] + self.scroll / 2, p[1]) for p in poly], color=(1, 1, 1))
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll:
                continue
            if poly[0][0] > self.scroll + VIEWPORT_W / SCALE:
                continue
            self.viewer.draw_polygon(poly, color=color)

        self.lidar_render = (self.lidar_render + 1) % 100
        i = self.lidar_render
        if i < 2 * len(self.lidar):
            l = self.lidar[i] if i < len(
                self.lidar) else self.lidar[len(self.lidar) - i - 1]
            self.viewer.draw_polyline(
                [l.p1, l.p2], color=(1, 0, 0), linewidth=1)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(
                        f.shape.radius, 30, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(
                        f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(
                        path, color=obj.color2, linewidth=2)

        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50 / SCALE
        x = TERRAIN_STEP * 3
        self.viewer.draw_polyline(
            [(x, flagy1), (x, flagy2)], color=(0, 0, 0), linewidth=2)
        f = [(x, flagy2), (x, flagy2 - 10 / SCALE),
             (x + 25 / SCALE, flagy2 - 5 / SCALE)]
        self.viewer.draw_polygon(f, color=(0.9, 0.2, 0))
        self.viewer.draw_polyline(f + [f[0]], color=(0, 0, 0), linewidth=2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
