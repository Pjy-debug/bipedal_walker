import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dqn import DQN, ReplayBuffer
from tqdm import tqdm
import matplotlib.pyplot as plt

import time
import numpy as np


import collections
import random
import torch.nn.functional as F

import torch
import numpy as np


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    capacity = 5000  # 经验池容量
    #修改学习率
    lr = 2e-3  # 学习率
    #lr = 1e-4
    gamma = 0.9  # 折扣因子
    epsilon = 0.9  # 贪心系数
    target_update = 200  # 目标网络的参数的更新频率
    batch_size = 256
    min_size = 300  # 经验池超过200后再训练
    return_list = []  # 记录每个回合的回报
    max_epoch = 200


    # 实例化经验池
    replay_buffer = ReplayBuffer(capacity, offline_data_path="/mnt/mnt1/tyy/data/dqn_episode_dataset_pos_reward0_new.npy")
    # 实例化DQN
    agent = DQN(initial_model = '/home/teamcommon/tyy/MyLander/Env_agent/stage2/model/trans_49.pt',
                learning_rate=lr,
                gamma=gamma,
                epsilon=epsilon,
                target_update=target_update,
                device=device,
            )

    # 训练模型
    for epoch in range(0,max_epoch):  # 100回合
        print('epoch:',epoch)
        
        if replay_buffer.size() > min_size:
                # 从经验池中随机抽样作为训练集
                s, a, r, ns, d,  ai, nai = replay_buffer.sample(batch_size)
                # 构造训练集
                # print(s.shape)
                transition_dict = {
                    'inputs': s,
                    'actions': a,
                    'next_inputs': ns,
                    'rewards': r,
                    'dones': d,
                    'all_inputs':ai,
                    'next_all_inputs':nai
                }
                # 网络更新
                agent.update(transition_dict)
    # 在这里添加保存模型的代码
    save_path = "reward0_stage3_final_model.pt"
    torch.save(agent.q_net.state_dict(), save_path)
    print(f"训练完成，模型已保存至: {save_path}")


if __name__ == '__main__':
    main()




