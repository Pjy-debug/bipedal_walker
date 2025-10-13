import numpy as np
from copy import deepcopy
import gymnasium as gym
#import torch
import math
import random
from scipy.stats import norm

"""
from tta.niches.box2d.env import make_env, bipedhard_custom
from tta.niches.box2d.bipedal_walker_test import BipedalWalkerTest, Env_config
"""

TERRAIN_STEP = 14/30
TERRAIN_GRASS = 10
GRASS = 0
STUMP = 1
PIT = 2
STATES = 4
#TERRAIN_HEIGHT = 200
#TERRAIN_STARTPAD = 20
#TERRAIN_LENGTH = 200
ground_roughness=0.6

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

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
model_path = '/root/brx/tta_new/tta/criticality/new_log/model/mlp.ep142'
print('loading model...')
criticality_model = torch.load(model_path)
criticality_model.to(device)
criticality_model.eval()
print("Successfully loading model!")


def calculate_criticality(terrain_state,states,cur_pos, terrain, current_i,agent_i, pred_i, terrain_counter, 
                          terrain_x, terrain_velocity, current_y, env_params, config, env_actions,np_random):
    """
    terrain_state: env.terrain_state
    
    model_path = 'new_log/model/mlp.ep142'
    print('loading model...')
    criticality_model = torch.load(model_path)
    criticality_model.eval()
    print("Successfully loading model!")
    """
    criticality = []
    p_orins = []
    weights = []
    q_list = []
    p_list = []
    weights = []
    polys = []
    if terrain_state == 0:
        
        for action in range(5,15):
            p_orin,q, cur_step_poly,terrain_num = calculate_criticality_(terrain_state,action, criticality_model,states,cur_pos,terrain, current_i,agent_i, pred_i, terrain_counter, terrain_x, terrain_velocity, current_y, env_params, config, env_actions,np_random)
            #q_list.append(q)
            #p_orins.append(p_orin)
            criticality.append(q * 0.1 * p_orin)
            """
            if q>0.4:
                criticality.append(q*0.1*p_orin)
            else:
                criticality.append(0)
            """
            p_list.append(0.1 * p_orin)
            polys.append(cur_step_poly)
            weights.append(0.1*p_orin/q)
        
        # this step q pdf
        criticality = np.array(criticality) / sum(criticality)
        #print(sum(criticality))
        #p_orins = np.array(p_orins) / sum(p_orins)
        p_list = np.array(p_list) / sum(p_list)
        #print(criticality)
        #print(p_orins)
        #print(p_list)
        pdf_array = epsilon_greedy(criticality, p_list, epsilon=0.05)
        #print(pdf_array)
        action_idx = np.random.choice(10, 1, p=pdf_array)
        action_idx = action_idx[0]
        #print(action_idx)
        #p_orin = p_orins[action_idx]
        p = p_list[action_idx]
        q = pdf_array[action_idx]
        poly = polys[action_idx]
        weight_1 = p * p_orin / q
        weight_2 = weights[action_idx]
        cur_action = action_idx + 5
        """
        cur_action = np.random.randint(5, 15)
        """
            
        assert cur_action > 4 and cur_action < 15
                    
    else:
    
        for action in range(5):
            
            p_orin,q, cur_step_poly,terrain_num = calculate_criticality_(terrain_state, action, criticality_model,states,cur_pos,terrain, current_i, agent_i,pred_i, terrain_counter, terrain_x, terrain_velocity, current_y, env_params, config, env_actions,np_random)
            #q_list.append(q)
            #p_orins.append(p_orin)
            criticality.append(q * 0.1 * p_orin)
            """
            if q>0.4:
                criticality.append(q*0.2*p_orin)
            else:
                criticality.append(0)
            """
            p_list.append(0.2*p_orin)
            polys.append(cur_step_poly)
            weights.append(0.2*p_orin/(q**terrain_num))
            
        criticality = np.array(criticality) / sum(criticality)
        #p_orins = np.array(p_orins) / sum(p_orins)
        p_list = np.array(p_list) / sum(p_list)
        #print(criticality)
        #print(p_orins)
        #print(p_list)
        pdf_array = epsilon_greedy(criticality, p_list, epsilon=0.05)
        #print(pdf_array)
        action_idx = np.random.choice(5, 1, p=pdf_array)
        action_idx = action_idx[0]
        #print(action_idx)
        #p_orin = p_orins[action_idx]
        p = p_list[action_idx]
        q = pdf_array[action_idx]
        poly = polys[action_idx]
        # 这个p_orin可以看作另一种权重
        weight_1 = p * p_orin / q
        weight_2 = weights[action_idx]
        cur_action = action_idx 
        """
        cur_action = np.random.randint(5)
    p_orin,q = calculate_criticality_(terrain_state,cur_action, criticality_model,states,cur_pos,terrain, current_i,agent_i, pred_i, terrain_counter, terrain_x, terrain_velocity, current_y, env_params, config, env_actions,np_random)
    """
    #print('p_orin:',p_orin,'q:',q,'weight:',weight,'cur_action:',cur_action)
    #print(criticality)
    #print(p_orins)
    return cur_action, weight_1 ,weight_2,p, q, poly
    #return cur_action,1,1,1,1,1
    
            

def calculate_criticality_(terrain_state, action, criticality_model,states,cur_pos,terrain, current_i,agent_i, pred_i, 
                           terrain_counter, terrain_x, terrain_velocity, current_y, env_params, config, env_actions,np_random):
    
    old_cur_i = current_i
    p_orin, cur_step_poly,terrain_num = env_step(terrain_state=terrain_state,action=action,terrain=terrain, current_i=current_i, 
                                     pred_i=pred_i, terrain_counter=terrain_counter, terrain_x=terrain_x,
                                    terrain_velocity=terrain_velocity, current_y=current_y, 
                                    env_params=env_params,config=config,np_random=np_random)
    
    if pred_i >= len(env_actions):
        pred_i = len(env_actions)-1
    #print(int(agent_i),agent_i>pred_i,agent_i,pred_i,len(env_actions))
    if int(agent_i) >= len(env_actions):
        agent_i = len(env_actions)-1
        #print('verify..',agent_i)
    pre_terrain_num = np.linspace(agent_i,pred_i,10,endpoint=True)
    pre_terrain_num = [int(item) for item in pre_terrain_num]
    pre_terrain = []
    #print(len(env_actions),pre_terrain_num)
    for n in range(len(pre_terrain_num)-1):
        pre_terrain_info = env_actions[pre_terrain_num[n]]
        next_terrain_info = env_actions[pre_terrain_num[n+1]]
        if pre_terrain_info['terrain_type'] == 1 or 'env_poly' not in pre_terrain_info:
            pre_terrain.append([(pre_terrain_info['start_x'],pre_terrain_info['current_y']),
                                (next_terrain_info['start_x'],next_terrain_info['current_y']),
                                (next_terrain_info['start_x'],next_terrain_info['current_y']),
                                (pre_terrain_info['start_x'],pre_terrain_info['current_y'])])
        else:
            pre_terrain.append(pre_terrain_info['env_poly'])
            
    terrain = []
    for x in pre_terrain:
        for item in x:
            terrain += list(item)
    for x in cur_step_poly:
        terrain += list(x)
    
    inputs = list(states) + list(cur_pos) + terrain + [action]
    #print(len(inputs))
    inputs = torch.tensor(inputs, dtype=torch.float32,device=device)
    inputs = inputs.reshape(1,-1)
    outputs,_ = criticality_model(inputs)
    #print(outputs[0])
    q = outputs[0][-1].item()
    #print(q)
    
    return p_orin, q, cur_step_poly,terrain_num

    
    
def env_step(terrain_state, action, terrain, current_i, pred_i, terrain_counter, terrain_x,
             terrain_velocity, current_y, env_params,config,np_random,hardcore=True, sparse_IS=True, debug=False):
    #print('-------------my_env_step!--------------','terrain_state=',terrain_state)
    #print('current_i=',current_i,'pred_i=',pred_i)
    """
    action : env_action
    terrain: env.terrain
    current_i: env.current_i
    pred_i: env.pred_i
    terrain_counter: env.terrain_counter
    terrain_x: env.terrain_x
    GRASS: self.GRASS
    STATES: self._STATES_
    terrain_velocity:self.terrain_velocity
    current_y: self.current_y
    env_params: self.env_params
    config: self.config
    """
    terrain.reverse()
    env_actions = []
    init_i = current_i
    cur_step_poly = []
    terrain_y = []
    p_orin = 1
    input_action = action
    first_terrain = True

    # if debug: print(f"init_i = {init_i}, pred_i = {pred_i}, pred_bound = {pred_bound}")
    # print('env take actions!')
    terrain_num = 0
    while current_i < pred_i or terrain_counter > 0:
    #while terrain_counter > 0:
        #print('current_i=',current_i,'pred_i=',pred_i)
        env_action = {}

        # 本次更新的开始x值
        x = current_i * TERRAIN_STEP
        terrain_x.append(x)
        current_i += 1
        
        """
        if (not cur_step_poly) and (not first_terrain) and terrain_counter == 0:
            break
        """
        if debug: print(f"i = {current_i}, x = {round(x, 2)}")
        #print('input_action:',action,'terrain_counter:',terrain_counter,'first_terrain:',first_terrain)
        
        if terrain_counter == 0:
            #print(f"self.terrain_state = {self.terrain_state}")
            # 草地
            if terrain_state == GRASS and hardcore:
                #print(f"action = {action},terrain_state={terrain_state}")
                if action <= 4:
                    if input_action < 10:
                        action += 5  
                    elif input_action >= 10:
                        action += 10
                assert action > 4 and action < 15
                if sparse_IS:
                    terrain_state = 1 if action > 9 else 2
                    terrain_num += 1
                    if first_terrain:
                        p_orin *= 2/3
                    
                    #print('terrain counter == 0 and terrain_state == grass, p=',p_orin,'action=',action,'first_terrain=',first_terrain)
                else:
                    # 下一个生成地形类别
                    terrain_state = np_random.integers(1, self._STATES_)
                if debug: print(f"self.terrain_state = {self.terrain_state}")
            else:
                # 坑、凸起后一定是草地
                #print(f"action = {action},terrain_state={terrain_state}")
                if action > 4 and action < 10:
                    action -= 5  # [0-4]
                elif action > 9:
                    action -= 10
                terrain_state = GRASS
                if sparse_IS:
                    # [5,10]
                    terrain_counter = action + 5
                    #p_orin *= 2 / TERRAIN_GRASS   # 10
                    #p = 1
                    terrain_num += 1
                    
                    if first_terrain:
                        p_orin *= 1/3
                    
                    else:
                        p_orin *= 1/2
                
                    #print('terrain counter == 0 and terrain_state != grass, p=',p_orin,'first_terrain=',first_terrain)
                else:
                    terrain_counter = np_random.integers(TERRAIN_GRASS / 2, TERRAIN_GRASS)
                if debug: print(f"self.terrain_counter = {terrain_counter}")
            terrain_oneshot = True


        if terrain_state == GRASS and not terrain_oneshot:
            terrain_velocity = 0.8 * terrain_velocity + \
                 0.01 * np.sign(TERRAIN_HEIGHT - current_y)
            if env_params is not None and env_params.altitude_fn is not None:
                print('env_params is not none!')
                current_y += terrain_velocity
                if i > TERRAIN_STARTPAD:
                    mid = TERRAIN_LENGTH * TERRAIN_STEP / 2.
                    x_ = (x - mid) * np.pi / mid
                    current_y = TERRAIN_HEIGHT + env_params.altitude_fn((x_, ))[0]
                    if i == TERRAIN_STARTPAD+1:
                        y_norm = env_params.altitude_fn((x_, ))[0]
                    current_y -= y_norm
                if debug: print(f"cppn used, i = {self.current_i}, y = {self.current_y}")
            else:
                if sparse_IS:
                    # action来控制草地摩擦程度
                    if action >=0 and action < 5:
                        delta_v = (action - 2 ) / 2
                    elif action >=5 and action < 15:
                        delta_v = (action - 9.5) / 4.5
                    terrain_velocity += delta_v  / SCALE
                    """
                    delta_v = np_random.uniform(-1, 1) / SCALE
                    terrain_velocity += delta_v 
                    p = 1/2
                    """
                    p = 1
                    """
                    if first_terrain:
                        p_orin *= p
                    """
                    #print('GRASS=',GRASS,'terrain counter > 0 and terrain_state == grass, p=',p,'action=',action,'delta_v=',delta_v,'first_terrain=',first_terrain)
                else:
                    delta_v = np_random.uniform(-1, 1) / SCALE
                    terrain_velocity += delta_v

                current_y += config.ground_roughness * terrain_velocity
                if debug: print(f"y = {round(current_y, 2)}")

        elif terrain_state == PIT and terrain_oneshot:
            if sparse_IS:
                # 用action来控制坑的宽度 [0,0.8]
                pit_gap = 1.0 + (action - 5) * 0.2
                # 这是一个固定值，不用再乘p，完全由action控制
                p = 1 / (config.pit_gap[1] - config.pit_gap[0])
                p_orin *= 1
                #print('pit=',PIT,'terrain counter == 0 and terrain_state == git, p=', p,'pit_gap=',pit_gap,'action=',action)
            else:
                pit_gap = 1.0 + np_random.uniform(*config.pit_gap)
            if debug: print(f"pit_gap = {pit_gap}")

            terrain_counter = np.ceil(pit_gap)
            pit_diff = terrain_counter - pit_gap

            poly = [
                (x,                current_y),
                (x + TERRAIN_STEP, current_y),
                (x + TERRAIN_STEP, current_y - 4 * TERRAIN_STEP),
                (x,                current_y - 4 * TERRAIN_STEP),
            ]
            terrain_counter += 2
            cur_step_poly = [(po[0] + TERRAIN_STEP * pit_gap, po[1]) for po in poly]
            original_y = current_y

        elif terrain_state == PIT and not terrain_oneshot:
            current_y = original_y
            if terrain_counter > 1:
                current_y -= 4 * TERRAIN_STEP
            if terrain_counter == 1:
                terrain_x[-1] = terrain_x[-1] - pit_diff * TERRAIN_STEP
                pit_diff = 0

        elif terrain_state == STUMP and terrain_oneshot:
            if sparse_IS:
                # 用action来控制桩的高度
                # 1
                stump_width = np_random.integers(*config.stump_width)
                # [0,0.4]
                stump_height = (9 - action) * 0.1
                if stump_height < 0:
                    stump_height = -stump_height
                # 0
                stump_float = np_random.integers(*config.stump_float)
                p = 1
                p_orin *= p
                #print('stump=',STUMP,'terrain counter == 0 and terrain_state == stump', 'p=',p,'width,height,float=',stump_width,stump_height,stump_float,'action=',action)
            else:
                stump_width = np_random.integers(*config.stump_width)
                stump_height = np_random.uniform(*config.stump_height)
                stump_float = np_random.integers(*config.stump_float)
            if debug:
                print(f"stump_width = {stump_width}, stump_height = {stump_height}, stump_float = {stump_float}")
            terrain_counter = stump_width
            countery = stump_height
            poly = [
                (x,                              current_y + stump_float * TERRAIN_STEP),
                (x + stump_width * TERRAIN_STEP, current_y + stump_float * TERRAIN_STEP),
                (x + stump_width * TERRAIN_STEP, current_y + countery *TERRAIN_STEP + stump_float * TERRAIN_STEP),
                (x,                              current_y + countery * TERRAIN_STEP + stump_float * TERRAIN_STEP),
            ]
            cur_step_poly = poly

        terrain_oneshot = False
        terrain_y.append(current_y)
        terrain_counter -= 1
        first_terrain = False

    # current_i = pred_i
    # print('start_i:',init_i,'end_i:',self.current_i,'total_step:',self.current_i-init_i)
    if not cur_step_poly:
        cur_step_poly = [(init_i*TERRAIN_STEP, current_y),(current_i * TERRAIN_STEP, current_y),
                         (init_i*TERRAIN_STEP, current_y),(current_i * TERRAIN_STEP, current_y)]
    #print('cur_step_poly',cur_step_poly)
    #print('p_orin',p_orin,'terrain_num',terrain_num)
    return p_orin, cur_step_poly, terrain_num

def epsilon_greedy(pdf_before_epsilon, ndd_pdf, epsilon=0.05):
    # NDD epsilon greedy method
    pdf_after_epsilon = (1 - epsilon) * pdf_before_epsilon + epsilon * ndd_pdf
    #assert (0.99999 <= np.sum(pdf_after_epsilon) <= 1.0001)
    return pdf_after_epsilon  

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
    
    
"""

class DeepCopyableWrapper(gym.Wrapper):

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memodict))
        return result
    
def bipedal_walker_criticality(i, actions, get_action):

    #Calculate criticality by Monte Carlo Sampling.

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
    
    #Calculate criticality by Monte Carlo Sampling.

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
"""