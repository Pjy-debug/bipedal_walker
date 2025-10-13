import sys, os

from .criticality_model import TransformerEncoder, Criticality_model_mlp, Criticality_model_trans,Reward_Model
from tqdm import tqdm
import matplotlib.pyplot as plt

import time
import numpy as np


import collections
import random
import torch.nn.functional as F

import torch
import numpy as np


def generate_seed():
    seed = os.getpid() + int(time.time() * 1e5 % 1e6)
    return seed


# 经验回放池
class ReplayBuffer():
    def __init__(self, capacity, is_offline = 1, offline_data_path='/mnt1/brx/dataset/dqn_episode_dataset_450.npy'):
        # 创建一个先进先出的队列，最大长度为capacity，保证经验池的样本量不变
        if is_offline:
            dqn_data = np.load(offline_data_path,allow_pickle=True)
            self.dqn_data = list(dqn_data)
            print(len(self.dqn_data))
            self.buffer = collections.deque(maxlen=len(self.dqn_data)+1000)
            for item in self.dqn_data:
                state = torch.tensor(item['input'])
                next_state = torch.tensor(item['next_input'])
                action = torch.tensor(item['action'])
                done = torch.tensor(item['done'])
                reward = torch.tensor(item['reward'])
                all_inputs = item['all_input']
                next_all_inputs = item['next_all_input']
                self.buffer.append((state, action, reward, next_state, done,  all_inputs, next_all_inputs))
        else:
            self.buffer = collections.deque(maxlen=capacity)
    # 将数据以元组形式添加进经验池
    def add(self, state, action, reward, next_state, done, all_inputs, next_all_inputs):
        self.buffer.append((state, action, reward, next_state, done, all_inputs, next_all_inputs))
    # 随机采样batch_size行数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)  # list, len=32
        # *transitions代表取出列表中的值，即32项
        state, action, reward, next_state, done,  all_input, next_all_input = zip(*transitions)
        #print(type(state[0]))
        #print(action)
        #print(reward)
        #print(terrain_state)
        #print(done)
        
        # print(torch.tensor(all_input[0]).shape)
        all_input = [ torch.tensor(item, dtype=torch.float) for item in all_input]
        next_all_input = [ torch.tensor(item, dtype=torch.float) for item in next_all_input]
        return torch.stack(list(state),dim=0), list(action), list(reward), torch.stack(list(next_state),dim=0), list(done), list(all_input), list(next_all_input)
    # 目前队列长度
    def size(self):
        return len(self.buffer)
    # offline

class DQN:
    def __init__(self, initial_model, learning_rate, gamma, epsilon, target_update, device):
        # 训练时的学习率
        self.learning_rate = learning_rate  
        # 折扣因子，对下一状态的回报的缩放
        self.gamma = gamma  
        # 贪婪策略，有1-epsilon的概率探索
        self.epsilon = epsilon  
        # 目标网络的参数的更新频率
        self.target_update = target_update 
        # 在GPU计算
        self.device = device  
        # 计数器，记录迭代次数
        self.count = 0
 
        # 构建2个神经网络，相同的结构，不同的参数
        self.q_net = Criticality_model_trans()
        #self.q_net.load_state_dict(torch.load('/root/autodl-tmp/model/mlp_model_253.pt')) #253
        self.q_net.load_state_dict(torch.load('/home/teamcommon/tyy/MyLander/Env_agent/stage2/model/trans_49.pt'))
        
        #self.q_net = torch.load(initial_model)
        
        """
        for name,param in self.q_net.named_parameters():
            
            if ('classifier' in name) or ('cls_head' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
            
            #print(name)
        """
        self.q_net.to(device)
        # print(self.q_net)
        #self.target_q_net = torch.load(initial_model)
        self.target_q_net = Criticality_model_trans()
        self.target_q_net.load_state_dict(torch.load('/home/teamcommon/tyy/MyLander/Env_agent/stage2/model/trans_49.pt'))
        self.target_q_net.to(device)
        
        self.t = 0
 
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
 
    def update(self, transition_dict): 
        # 获取当前时刻的状态 array_shape=[b,4]
        """
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        # 获取当前时刻采取的动作 tuple_shape=[b]，维度扩充 [b,1]
        actions = torch.tensor(transition_dict['actions']).view(-1,1)
        """
        # 网络输入
        # print(inputs.shape)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        inputs = torch.tensor(transition_dict['inputs'], dtype=torch.float).to(device)
        # print(inputs.shape)
    
        # 什么动作
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(device)
        # 当前状态下采取动作后得到的奖励 tuple=[b]，维度扩充 [b,1] --> reward model + 原来
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(device)
        
        """
        # 下一时刻的状态 array_shape=[b,4]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        """
        next_inputs = torch.tensor(transition_dict['next_inputs'], dtype=torch.float).to(device)
        
        # 是否到达目标 tuple_shape=[b]，维度变换[b,1]
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(device)
        # terrain_state
        # next_all_inputs = torch.tensor(transition_dict['next_all_inputs'], dtype=torch.float).to(device)
        next_all_inputs = transition_dict['next_all_inputs']
 
        # 输入当前状态，得到采取各运动得到的奖励 [b,4]==>[b,2]==>[b,1]
        # print(inputs.shape)
        outputs, _, _, _ = self.q_net(inputs, inputs, 0.5)
        q_values = outputs[0][-1]
        # 下一时刻的状态[b,4]-->目标网络输出下一时刻对应的动作q值[b,2]-->
        # 选出下个状态采取的动作中最大的q值[b]-->维度调整[b,1]
        
        max_next_q_values = []
        #print(inputs.shape[0])
        for k in range(inputs.shape[0]):
            """
            new_next_inputs = []
            # print(terrain_state[k])
            if terrain_state[k] == 0:
                for action in range(5,15):
                    new_next_input = torch.cat((next_inputs[k,:-1], torch.tensor([action]).to(device)), dim=0)
                    new_next_inputs.append(new_next_input)
            else:
                for action in range(5):
                    new_next_input = torch.cat((next_inputs[k,:-1], torch.tensor([action]).to(device)), dim=0)
                    new_next_inputs.append(new_next_input)
            new_next_inputs = torch.stack(new_next_inputs,dim=0)
            # print(new_next_inputs.shape)
            """
            new_next_inputs = next_all_inputs[k].to(device)
            #print(new_next_inputs.shape)
            next_outputs, _, _, _ = self.target_q_net(new_next_inputs, new_next_inputs, 0.5)
            # print(next_outputs.shape)
            max_next_q_value, _ = torch.max(next_outputs[:,-1],dim=0)
            max_next_q_values.append(max_next_q_value.item())
        # 目标网络输出的当前状态的q(state_value)：即时奖励+折扣因子*下个时刻的最大回报
        #print('rewards:',rewards)
        max_next_q_values = torch.tensor(max_next_q_values).reshape(-1,1).to(device)
        #print('max_q:',max_next_q_values)
        q_targets = rewards + self.gamma * max_next_q_values * (1-dones)
        #print('q_targets:',q_targets)
 
        # 目标网络和训练网络之间的均方误差损失
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        print('dqn_loss:',dqn_loss.item())
        # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        self.optimizer.zero_grad()
        # 反向传播参数更新
        dqn_loss.backward()
        # 对训练网络更新
        self.optimizer.step()
        
        torch.save(self.q_net.state_dict(),f'model_stage3_only/dqn_model_mlp_450_{self.t}.pt')
        self.t += 1
 
        # 在一段时间后更新目标网络的参数
        if self.count % self.target_update == 0:
            # 将目标网络的参数替换成训练网络的参数
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())
        
        self.count += 1
    '''
    def update(self, transition_dict): 
        # 获取当前时刻的状态 array_shape=[b,4]
        """
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        # 获取当前时刻采取的动作 tuple_shape=[b]，维度扩充 [b,1]
        actions = torch.tensor(transition_dict['actions']).view(-1,1)
        """
        # 网络输入
        # print(inputs.shape)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        inputs = torch.tensor(transition_dict['inputs'], dtype=torch.float).to(device)
        # print(inputs.shape)
    
        # 什么动作
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(device)
        # 当前状态下采取动作后得到的奖励 tuple=[b]，维度扩充 [b,1] --> reward model + 原来
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(device)
        
        """
        # 下一时刻的状态 array_shape=[b,4]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        """
        next_inputs = torch.tensor(transition_dict['next_inputs'], dtype=torch.float).to(device)
        
        # 是否到达目标 tuple_shape=[b]，维度变换[b,1]
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(device)
        # terrain_state
        # next_all_inputs = torch.tensor(transition_dict['next_all_inputs'], dtype=torch.float).to(device)
        next_all_inputs = transition_dict['next_all_inputs']
 
        #输入当前状态，得到采取各运动得到的奖励 [b,4]==>[b,2]==>[b,1]
        print(inputs.shape)
        outputs, _, _, _ = self.q_net(inputs, inputs, 0.5)

        #q_values = outputs[0][-1]
        # 下一时刻的状态[b,4]-->目标网络输出下一时刻对应的动作q值[b,2]-->
        # 选出下个状态采取的动作中最大的q值[b]-->维度调整[b,1]
        
        max_next_q_values = []
        #print(inputs.shape[0])
        for k in range(inputs.shape[0]):
            """
            new_next_inputs = []
            # print(terrain_state[k])
            if terrain_state[k] == 0:
                for action in range(5,15):
                    new_next_input = torch.cat((next_inputs[k,:-1], torch.tensor([action]).to(device)), dim=0)
                    new_next_inputs.append(new_next_input)
            else:
                for action in range(5):
                    new_next_input = torch.cat((next_inputs[k,:-1], torch.tensor([action]).to(device)), dim=0)
                    new_next_inputs.append(new_next_input)
            new_next_inputs = torch.stack(new_next_inputs,dim=0)
            # print(new_next_inputs.shape)
            """
            new_next_inputs = next_all_inputs[k].to(device)
            print(new_next_inputs.shape)
            next_outputs, _, _, _ = self.target_q_net(new_next_inputs, new_next_inputs, 0.5)

            # print(next_outputs.shape)
            #max_next_q_value, _ = torch.max(next_outputs[:,-1],dim=0)
            max_next_q_values.append(max_next_q_value.item())
        # 目标网络输出的当前状态的q(state_value)：即时奖励+折扣因子*下个时刻的最大回报
        #print('rewards:',rewards)
        max_next_q_values = torch.tensor(max_next_q_values).reshape(-1,1).to(device)
        #print('max_q:',max_next_q_values)
        q_targets = rewards + self.gamma * max_next_q_values * (1-dones)
        #print('q_targets:',q_targets)
 
        # 目标网络和训练网络之间的均方误差损失
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        print('dqn_loss:',dqn_loss.item())
        # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        self.optimizer.zero_grad()
        # 反向传播参数更新
        dqn_loss.backward()
        # 对训练网络更新
        self.optimizer.step()
        
        torch.save(self.q_net.state_dict(),f'model_stage3_only/dqn_model_mlp_450_{self.t}.pt')
        self.t += 1
 
        # 在一段时间后更新目标网络的参数
        if self.count % self.target_update == 0:
            # 将目标网络的参数替换成训练网络的参数
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())
        
        self.count += 1
    '''



