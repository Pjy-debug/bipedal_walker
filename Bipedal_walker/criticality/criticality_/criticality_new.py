'''*************************************************************************
[file name]                 criticality_new.py
[function details]          用于测试过程中计算Criticality的相关函数, Criticality_model类和Reward_model类等
                            在criticality_model.py中
[developer]                 （必需）
[change log]               （若修改过则必需注明）
*************************************************************************'''
import sys
import os
# 获取当前文件的绝对路径
current_path = os.path.abspath(__file__)
# 向上一级目录
parent_dir = os.path.dirname(os.path.dirname(current_path))
# 将criticality目录添加到Python的模块搜索路径
sys.path.append(parent_dir)

import numpy as np
from copy import deepcopy
import gymnasium as gym
import torch
import math
import random
from scipy.stats import norm
from env_step import env_step
# 不知道为什么，这里不从父文件夹导入的话nade_test就会找不到criticality_model.py
from criticality_.criticality_model import Criticality_model_mlp,Criticality_model, distill_q_trans, Reward_Model



"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#'/root/brx/tta_new/tta/criticality/new_log/model/mlp.ep142/939'
# 39 199 299 248
model_path = '/root/brx/tta_new/tta/criticality/new_log/model_new/mlp.ep546'
print('loading model...')
criticality_model = torch.load(model_path)
criticality_model.to(device)
criticality_model.eval()
print("Successfully loading model!")
"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'
rw=Reward_Model()
rw.load_state_dict(torch.load('/home/teamcommon/pjy/Bipedal_walker/criticality/stage1/storage/rw_149.pt' ))
rw.eval().to(device)

cm =  Criticality_model()
#cm.load_state_dict(torch.load('/root/autodl-tmp/new_nde/new_model/stage2/trans_197.pt')) #99
cm.load_state_dict(torch.load('/home/teamcommon/pjy/Bipedal_walker/criticality/stage3/dqn_model_2.pt' ))

cm.eval().to(device)



def dqn_collect_data(terrain_state,states,cur_pos, terrain, current_i,agent_i, pred_i, terrain_counter, 
                          terrain_x, terrain_velocity, current_y, env_params, config, env_actions,np_random):
    all_inputs = []
    if terrain_state == 0:
        for action in range(5,15):
            #print(env_actions)
            p_orin,q, cur_step_poly,terrain_num, inputs = calculate_criticality_(terrain_state,action,states,cur_pos,current_i,agent_i, pred_i, terrain_counter, terrain_x, terrain_velocity, current_y, env_params, config, env_actions,np_random)
            all_inputs.append(inputs)
        cur_action = np.random.randint(5, 15)
        cur_inputs = all_inputs[cur_action-5]
    
                    
    else:
    
        for action in range(5):
            #print(env_actions)
            p_orin,q, cur_step_poly,terrain_num, inputs = calculate_criticality_(terrain_state, action,states,cur_pos, current_i, agent_i,pred_i, terrain_counter, terrain_x, terrain_velocity, current_y, env_params, config, env_actions,np_random)
            all_inputs.append(inputs)
        cur_action = np.random.randint(5)
        cur_inputs = all_inputs[cur_action]
      
    return cur_action, cur_inputs, all_inputs

'''*************************************************************************
[function name]                 （必需）
[function details]          计算Criticality的函数; Use Reward_Model('rw_149.pt') and 
                            Criticality_model('dqn_model_2.pt') to calculate criticality.
[inputs]                    int | terrain_state: 
                            list | states: lidar.fraction
                            tuple | cur_pos: x,y
                            None | terrain
                            int | current_i
                            int | agent_i
                            int | pred_i
                            int | terrain_counter
                            list | terrain_x: len is TERRAIN_LENGTH (200)
                            float | terrain_velocity
                            float | current_y
                             | env_params
                            Env_config | config
                            list | env_actions: list of env actions (int)
                            np.random.RandomState | np_random: seeding.np_random(seed)
[outputs]
[developer&date]           （必需）
[change log]               （若修改过则必需注明）
*************************************************************************'''
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
            #print(env_actions)
            
            p_orin,q, cur_step_poly,terrain_num,_ = calculate_criticality_(terrain_state,action,states,cur_pos,current_i,agent_i, pred_i, terrain_counter, terrain_x, terrain_velocity, current_y, env_params, config, env_actions,np_random)
            
            criticality.append(q)
            p_list.append(0.1)
            #polys.append(cur_step_poly)
            #weights.append(0.1*p_orin/(q**terrain_num))
        
        # this step q pdf
        
        #print(criticality)
        if sum(criticality):
            criticality = np.array(criticality) / sum(criticality)
            #print(criticality)
            p_list = np.array(p_list) / sum(p_list)
            # 0.075
            pdf_array = epsilon_greedy(criticality, p_list, epsilon=0.05)
        else:
            pdf_array = p_list
        
        
        action_idx = np.random.choice(10, 1, p=pdf_array)
        action_idx = action_idx[0]
        
        p = p_list[action_idx]
        q = pdf_array[action_idx]
        c = criticality[action_idx]
        #poly = polys[action_idx]
        
        # 就是认为少乘了权重
        weight_1 = p  / (q) 
        """
        if not terrain_num:
            weight_1 = 1
        else:
            weight_1 = p  / (q) 
        """
        
        #print(action_idx)
        cur_action = action_idx + 5
        """
        cur_action = np.random.randint(5, 15)
        """
            
        assert cur_action > 4 and cur_action < 15
                    
    else:
    
        for action in range(5):
            #print(env_actions)
            p_orin,q, cur_step_poly,terrain_num,_ = calculate_criticality_(terrain_state, action,states,cur_pos, current_i, agent_i,pred_i, terrain_counter, terrain_x, terrain_velocity, current_y, env_params, config, env_actions,np_random)
            
            criticality.append(q)
            """
            if q>0.4:
                criticality.append(q*0.2*p_orin)
            else:
                criticality.append(0)
            """
            p_list.append(0.2)
            #polys.append(cur_step_poly)
        
        if sum(criticality):
            criticality = np.array(criticality) / sum(criticality)
            #print(criticality)
            p_list = np.array(p_list) / sum(p_list)
            # 0.075
            pdf_array = epsilon_greedy(criticality, p_list, epsilon=0.05)
        else:
            pdf_array = p_list
        
        
        #print(pdf_array)
        action_idx = np.random.choice(5, 1, p=pdf_array)
        action_idx = action_idx[0]
        #print(action_idx)
        #p_orin = p_orins[action_idx]
        p = p_list[action_idx]
        q = pdf_array[action_idx]
        c = criticality[action_idx]
        #poly = polys[action_idx]
        
        # 认为少乘了权重
        weight_1 = p  / (q) 
        
        """
        if not terrain_num:
            weight_1 = 1
        else:
            weight_1 = p / (q)
        """ 
        
        cur_action = action_idx 
        """
        cur_action = np.random.randint(5)
    p_orin,q = calculate_criticality_(terrain_state,cur_action, criticality_model,states,cur_pos,terrain, current_i,agent_i, pred_i, terrain_counter, terrain_x, terrain_velocity, current_y, env_params, config, env_actions,np_random)
    """
    #print('p=',p,'q=',q,'weight:',weight_1,'cur_action:',cur_action)
    #print(criticality)
    #print(p_orins)
    #print(pdf_array)
    return cur_action, weight_1 ,p, q, c
    #return cur_action,1,1,1,1,1
    
            
'''*************************************************************************
[function name]             calculate_criticality_
[function details]          计算Criticality的函数
[inputs]                    terrain_state, states, cur_pos, terrain, current_i, agent_i, 
                            pred_i, terrain_counter, terrain_x,terrain_velocity, current_y,
                            env_params, config, env_actions, np_random: 同上
                            sub detail: cur_pos: not used; terrain: not used;
                            int | actions: current action. 0-5 or 5-15
[outputs]
[developer&date]           （必需）
[change log]               （若修改过则必需注明）
*************************************************************************'''
def calculate_criticality_(terrain_state, action ,states,cur_pos, current_i,agent_i, pred_i, 
                           terrain_counter, terrain_x, terrain_velocity, current_y, env_params, config, env_actions,np_random):
    
    old_cur_i = current_i
   
    p_orin, cur_step_poly,terrain_num = env_step(my_terrain_state=terrain_state,my_action=action,my_current_i=current_i, my_pred_i=pred_i, my_terrain_counter=terrain_counter, terrain_x=terrain_x,
                                    my_terrain_velocity=terrain_velocity, my_current_y=current_y, 
                                    env_params=env_params,config=config,np_random=np_random)
    
    if pred_i >= len(env_actions):
        pred_i = len(env_actions)-1
    
    if int(agent_i) >= len(env_actions):
        agent_i = len(env_actions)-1
        #print('verify..',agent_i)
    pre_terrain_num = np.linspace(agent_i,pred_i,10,endpoint=True)
    pre_terrain_num = [int(item) for item in pre_terrain_num]
    
    #print('4',states)
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
    
    new_terrain = []
    for x in pre_terrain:
        for item in x:
            new_terrain += list(item)

    for x in cur_step_poly:
        new_terrain += list(x)
   
    inputs = list(states) + new_terrain + [action] # 输入状态，交给rm模型和cm模型


    inputs = torch.tensor(inputs, dtype=torch.float32,device=device)
    inputs = inputs.reshape(1,-1)
    # shape of inputs is (1, 105)

    with torch.no_grad():
        re = rw(inputs)
        outputs2, feats2, _, _ = cm(inputs, inputs, 0.5)
        pred = outputs2[:,-1]>0.1
        rewards = re['rewards']
        reward = rewards[0,-1].item()
        #if reward < 4.5:
        if reward < 3:
            q = 0
        else:
            q = outputs2[0,-1].item()
    #outputs,_,_,_ = criticality_model(inputs,inputs,0.5)
    #print(outputs[0])
    #q = outputs[0][-1].item()
    #print(q)
    #q = 1
    
    """
    p_orin = 0.1
    cur_step_poly = None
    terrain_num =0
    q = 0.1
    """
    
    return p_orin, q, cur_step_poly,terrain_num,inputs



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
    
    
