import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import math
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader,Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random

from utils.criticality_model import Reward_Model
from utils.data_utils import create_episode_dataset
from utils.reward_model import Trainer

import argparse
import matplotlib.pyplot as plt

                          
def main(args):
    print(args)
    writer = SummaryWriter(log_dir=args.log_dir)
    train_info = []
    val_info = []
    test_info = []
    
    
    # build dataset
    #记得换一下路径
    dataset_path_pos = '/mnt/mnt1/tyy/data/positive/train_pos.npy'
    dataset_path_pos1 = '/mnt/mnt1/tyy/data/positive/val_pos.npy'
    train_data_pos = np.load(dataset_path_pos, allow_pickle=True)
    train_data_pos = list(train_data_pos)
    train_dataset_pos = create_episode_dataset(train_data_pos, [], 'pos')
    
    val_data_pos = np.load(dataset_path_pos1, allow_pickle=True)
    val_data_pos = list(val_data_pos)
    val_dataset_pos = create_episode_dataset(val_data_pos,[],'pos')
    
    
    #记得换路径
    dataset_path_neg = '/mnt/mnt1/tyy/data/newnegative/train_neg.npy'
    train_data_neg = np.load(dataset_path_neg, allow_pickle=True)
    train_data_neg = list(train_data_neg)  
    
    dataset_path_neg1 = '/mnt/mnt1/tyy/data/newnegative/val_neg.npy'
    val_data_neg = np.load(dataset_path_neg1, allow_pickle=True)   
    val_data_neg = list(val_data_neg)
    

    train_dataset_neg = create_episode_dataset([],train_data_neg,'neg')
    
    val_dataset_neg = create_episode_dataset([],val_data_neg,'neg')
    
    #print(f'len is {len(train_dataset_neg)}')

    # --- 新增的代码: 选取 10% 的正样本数据 ---

    num_pos_samples = len(train_dataset_pos)
    subset_size = int(num_pos_samples * 0.1)
    
    # 随机生成索引
    np.random.seed(42) # 为了可复现性，设置随机种子
    indices = np.random.choice(range(num_pos_samples), size=subset_size, replace=False)
    
    # 创建正样本子集
    train_dataset_pos_subset = Subset(train_dataset_pos, indices)
    # --- 结束新增代码 ---

    # Build DataLoader
    train_pos_loader = DataLoader(train_dataset_pos_subset,batch_size=args.batch_size,shuffle=True,drop_last = True)
    train_neg_loader = DataLoader(train_dataset_neg,batch_size=args.batch_size,shuffle=True,drop_last = True)
    
    #新增
    # --- 验证集: 选取 10% 的正样本数据 ---
    num_pos_samples_val = len(val_dataset_pos)
    subset_size_val = int(num_pos_samples_val * 0.1)
    
    # 随机生成索引
    np.random.seed(42) # 同样为了可复现性
    indices_val = np.random.choice(range(num_pos_samples_val), size=subset_size_val, replace=False)
    
    # 创建正样本子集
    val_dataset_pos_subset = Subset(val_dataset_pos, indices_val)
    # --- 验证集部分结束 ---

    val_pos_loader = DataLoader(val_dataset_pos_subset, batch_size=64, shuffle=True)
    val_neg_loader = DataLoader(val_dataset_neg, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)


    # Build Trainer
    trainer = Trainer(args,train_pos_loader,train_neg_loader,val_pos_loader,val_neg_loader)
    print("trainer")

    #新增
    best_val_acc = -10.0
    # Train & Validate
    old_data_idx = 0
    for epoch in range(args.start_epoch,args.epochs):
        train_neg_loader = DataLoader(train_dataset_neg,batch_size=args.batch_size,shuffle=True,drop_last = True)
        trainer.train_neg_loader = train_neg_loader
        
        train_scores, train_acc = trainer.train(epoch)
        val_scores, val_acc = trainer.validate(epoch)
        if val_acc > best_val_acc:
            print(f'Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}. Saving model...')
            best_val_acc = val_acc
            # 修改 3: 调用保存函数，并使用一个新的文件名来区分最佳模型
            trainer.save(epoch, filename=f'40new_rw_model_f5_best.pt', root='/home/teamcommon/tyy/MyLander/Env_agent/stage1')
            #trainer.save(epoch, args.output_model_prefix) 原来的'''
    writer.close()

def run_test_with_threshold(args, threshold, model_stage1, model_stage2, test_loader, device):
    """
    运行带有特定阈值的双阶段测试，并返回性能指标。
    """
    print(f"\n--- Testing with threshold = {threshold:.2f} ---")
    
    # 初始化统计变量
    FP_count, FN_count, TP_count, TN_count = 0, 0, 0, 0
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            labels = labels.reshape(-1, 2)
            labels_cls = torch.tensor(labels[:, 0], dtype=torch.int64)
            bs = inputs.shape[0]

            # Stage 1 预测
            outputs_stage1 = model_stage1(inputs)
            rewards_stage1 = outputs_stage1["rewards"][:, -1]
            
            # Stage 1 预测标签
            pred_stage1 = (rewards_stage1 > threshold).cpu().numpy()

            # 找到所有 Stage 1 预测为正样本的索引
            stage1_pos_indices = np.where(pred_stage1 == 1)[0]
            
            final_predictions = pred_stage1.copy()

            if len(stage1_pos_indices) > 0:
                stage1_pos_inputs = inputs[stage1_pos_indices]
                outputs_stage2, _, _, _ = model_stage2(stage1_pos_inputs, stage1_pos_inputs, 0.5)
                rewards_stage2 = outputs_stage2[:, -1]
                pred_stage2 = (rewards_stage2 > args.threshold).cpu().numpy()
                
                # 检查形状是否匹配，并进行赋值
                if len(stage1_pos_indices) == (len(pred_stage2)//2):
                    final_predictions[stage1_pos_indices] = pred_stage2[:len(pred_stage2)//2]
                else:
                    print(f"Warning: Shape mismatch. Stage1 indices: {len(stage1_pos_indices)}, Stage2 predictions: {len(pred_stage2)}")
                    # 这里的处理方式取决于你的需求，如果只是想看结果，可以跳过这个批次
                    continue
            
            # 统计最终结果
            for k in range(bs):
                pred = final_predictions[k]
                label = labels_cls[k].item()
                
                if pred == 1 and label == 1:
                    TP_count += 1
                elif pred == 1 and label == 0:
                    FP_count += 1
                elif pred == 0 and label == 1:
                    FN_count += 1
                elif pred == 0 and label == 0:
                    TN_count += 1

    # 计算并打印性能指标
    total_pos = TP_count + FN_count
    total_neg = FP_count + TN_count
    
    precision = TP_count / (TP_count + FP_count) if (TP_count + FP_count) > 0 else 0
    recall = TP_count / (TP_count + FN_count) if (TP_count + FN_count) > 0 else 0

    print(f"TP = {TP_count}, FN = {FN_count}, FP = {FP_count}, TN = {TN_count}")
    print(f"Final Precision = {precision:.4f}")
    print(f"Final Recall = {recall:.4f}")
    
    return precision, recall

def plot_final_scores(all_scores, all_labels):
    """
    绘制最终预测分数的散点图，并根据真实标签区分颜色。
    
    Args:
        all_scores (list): 所有样本的最终预测分数。
        all_labels (list): 所有样本的真实标签 (0或1)。
    """
    # 将列表转换为 NumPy 数组以便处理
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # 过滤出正样本和负样本的分数
    pos_scores = all_scores[all_labels == 1]
    neg_scores = all_scores[all_labels == 0]

    plt.figure(figsize=(14, 8))

    # 绘制负样本分数
    plt.scatter(range(len(neg_scores)), 
                neg_scores,  
                alpha=0.7, 
                marker='^', 
                c='dodgerblue', 
                label='Negative Samples')
    
    # 绘制正样本分数
    plt.scatter(range(len(pos_scores)), 
                pos_scores,  
                alpha=0.7, 
                marker='o', 
                c='coral', 
                label='Positive Samples')

    plt.title('Final Prediction Scores by True Label')
    plt.ylabel('Score')
    plt.xlabel('Sample Index')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/teamcommon/tyy/MyLander/Env_agent/stage1/final_scores_plot.png')
    plt.close()
    print("Final scores plot saved to final_scores_plot.png")
def evaluate_with_thresholds(all_scores, all_labels, min_threshold, max_threshold):
    """
    遍历不同的阈值，并计算模型的准确率、召回率等性能指标。

    Args:
        all_scores (list or np.array): 所有预测分数的列表或数组。
        all_labels (list or np.array): 所有真实标签的列表或数组。
        min_threshold (float): 阈值范围的最小值。
        max_threshold (float): 阈值范围的最大值。
    """
    print("\n--- Evaluating different thresholds ---")
    
    thresholds = np.arange(min_threshold, max_threshold + 0.01, 0.01)
    
    for threshold in thresholds:
        # 基于当前阈值生成预测标签
        pred_labels = (np.array(all_scores) > threshold).astype(int)
        
        # 计算真阳性、真阴性、假阳性、假阴性
        TP = np.sum((pred_labels == 1) & (np.array(all_labels) == 1))
        TN = np.sum((pred_labels == 0) & (np.array(all_labels) == 0))
        FP = np.sum((pred_labels == 1) & (np.array(all_labels) == 0))
        FN = np.sum((pred_labels == 0) & (np.array(all_labels) == 1))

        # 避免除以零
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        accuracy = (TP + TN) / len(all_labels) if len(all_labels) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FNR = FN / (FN + TP) if (FN + TP) > 0 else 0

        # 打印结果
        print(f"Threshold: {threshold:.2f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall (TPR): {recall:.4f}")
        print(f"  FPR: {FPR:.4f}")
        print(f"  FNR: {FNR:.4f}")
        print("-" * 20)
def run_two_stage_test(args, model_stage1, model_stage2, test_loader, device, stage1_threshold, stage2_threshold):
    """
    运行带有特定阈值的双阶段测试，并返回性能指标。
    """
    print(f"\n--- Testing with Stage 1 threshold = {stage1_threshold:.2f}, Stage 2 threshold = {stage2_threshold:.2f} ---")
    
    # 初始化统计变量
    FP_count, FN_count, TP_count, TN_count = 0, 0, 0, 0
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            labels = labels.reshape(-1, 2)
            labels_cls = torch.tensor(labels[:, 0], dtype=torch.int64)
            bs = inputs.shape[0]

            # Stage 1 预测
            outputs_stage1 = model_stage1(inputs)
            rewards_stage1 = outputs_stage1["rewards"][:, -1]
            
            # Stage 1 预测标签
            pred_stage1 = (rewards_stage1 > stage1_threshold).cpu().numpy()

            # 找到所有 Stage 1 预测为正样本的索引
            stage1_pos_indices = np.where(pred_stage1 == 1)[0]
            
            final_predictions = pred_stage1.copy()

            if len(stage1_pos_indices) > 0:
                stage1_pos_inputs = inputs[stage1_pos_indices]
                outputs_stage2, _, _, _ = model_stage2(stage1_pos_inputs, stage1_pos_inputs, 0.5)
                rewards_stage2 = outputs_stage2[:, -1]
                
                # 根据 Stage 2 的阈值进行判断
                pred_stage2 = (rewards_stage2 > stage2_threshold).cpu().numpy()
                
                # 检查形状是否匹配，并进行赋值
                if len(stage1_pos_indices) == len(pred_stage2)//2:
                    final_predictions[stage1_pos_indices] = pred_stage2[:len(pred_stage2)//2]
                else:
                    print(f"Warning: Shape mismatch. Stage1 indices: {len(stage1_pos_indices)}, Stage2 predictions: {len(pred_stage2)}. Skipping this batch.")
                    continue
            
            # 统计最终结果
            for k in range(bs):
                pred = final_predictions[k]
                label = labels_cls[k].item()
                
                if pred == 1 and label == 1:
                    TP_count += 1
                elif pred == 1 and label == 0:
                    FP_count += 1
                elif pred == 0 and label == 1:
                    FN_count += 1
                elif pred == 0 and label == 0:
                    TN_count += 1

    # 计算并打印性能指标
    total_pos = TP_count + FN_count
    total_neg = FP_count + TN_count
    
    precision = TP_count / (TP_count + FP_count) if (TP_count + FP_count) > 0 else 0
    recall = TP_count / (TP_count + FN_count) if (TP_count + FN_count) > 0 else 0

    print(f"TP = {TP_count}, FN = {FN_count}, FP = {FP_count}, TN = {TN_count}")
    print(f"Final Precision = {precision:.4f}")
    print(f"Final Recall = {recall:.4f}")
    
    return precision, recall 

# def test(args):
#     device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
#     random_seed = 42  
#     random.seed(random_seed)

#     # 加载测试数据集，这里省略了数据加载部分，保持原样
#     dataset_path_pos = '/mnt/mnt1/tyy/data/positive/test_pos.npy'
#     train_data_pos = np.load(dataset_path_pos, allow_pickle=True)
#     train_data_pos = list(train_data_pos)
#     random.shuffle(train_data_pos)
#     train_data_pos = train_data_pos[:len(train_data_pos) // 10]
    
#     dataset_path_neg = '/mnt/mnt1/tyy/data/newnegative/test_neg.npy'
#     train_data_neg = np.load(dataset_path_neg, allow_pickle=True)
#     train_data_neg = list(train_data_neg)
    
#     test_dataset = create_episode_dataset(train_data_pos, train_data_neg,'total')
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
#     # 步骤 1: 加载两个模型
#     model_stage1 = Reward_Model()
#     model_stage1.load_state_dict(torch.load('/home/teamcommon/tyy/MyLander/Env_agent/stage1/40new_rw_model_f5_best.pt'))
    
#     # 注意：Stage 2 的模型是 Criticality_model_trans
#     from utils.criticality_model import Criticality_model_trans
#     model_stage2 = Criticality_model_trans()
#     model_stage2.load_state_dict(torch.load('/home/teamcommon/tyy/MyLander/Env_agent/stage3/reward0_stage3_final_model.pt'))

#     model_stage1.eval().to(device)
#     model_stage2.eval().to(device)
#     # 定义要测试的Stage 1阈值
#     stage1_threshold = 0.1 
#     # 定义要测试的Stage 2阈值范围
#     stage2_thresholds = np.arange(0, 1.05, 0.05)
    
#     # # 遍历每个Stage 2阈值并进行测试
#     # for threshold in stage2_thresholds:
#     #     run_two_stage_test(args, model_stage1, model_stage2, test_loader, device, stage1_threshold, threshold)
        
#     # return 

    
#     # 初始化统计变量
#     FP_count, FN_count, TP_count, TN_count = 0, 0, 0, 0
    
#     with torch.no_grad():
#         for i, data in enumerate(test_loader):
#             inputs, labels = data[0].to(device), data[1].to(device)
#             labels = labels.reshape(-1, 2)
#             labels_cls = torch.tensor(labels[:, 0], dtype=torch.int64)
#             bs = inputs.shape[0]

#             # 步骤 2: 使用 Stage 1 模型进行预测
#             outputs_stage1 = model_stage1(inputs)
#             rewards_stage1 = outputs_stage1["rewards"][:, -1]
            
#             # Stage 1 的预测标签
#             pred_stage1 = (rewards_stage1 > 0.1).cpu().numpy()

#             # 步骤 3 & 4: 筛选 Stage 1 预测为正样本的所有数据，交给 Stage 2 重新预测
#             # 找到所有 Stage 1 预测为正样本的索引
#             stage1_pos_indices = np.where(pred_stage1 == 1)[0]
            
#             # 拷贝 Stage 1 的预测结果作为初始的最终预测结果
#             final_predictions = pred_stage1.copy()

#             if len(stage1_pos_indices) > 0:
#                 # 获取 Stage 1 预测为正样本的输入数据
#                 stage1_pos_inputs = inputs[stage1_pos_indices]
                
#                 # `inputs` 和 `inputs` 传入 Stage 2 模型
#                 outputs_stage2, _, _, _ = model_stage2(stage1_pos_inputs, stage1_pos_inputs, 0.5)
                
#                 # 获取 Stage 2 的预测分数
#                 rewards_stage2 = outputs_stage2[:, -1]
                
#                 # 根据 Stage 2 的预测，判断这些样本的最终分类
#                 pred_stage2 = (rewards_stage2 > 0.5).cpu().numpy()
                
#                 # 步骤 5: 更新 Stage 1 预测结果，用 Stage 2 的结果覆盖
#                 # 仅更新那些 Stage 1 预测为正样本的样本
#                 final_predictions[stage1_pos_indices] = pred_stage2[:len(pred_stage2)//2]
            
#             # 统计最终结果
#             for k in range(bs):
#                 pred = final_predictions[k]
#                 label = labels_cls[k].item()
                
#                 if pred == 1 and label == 1:
#                     TP_count += 1
#                 elif pred == 1 and label == 0:
#                     FP_count += 1
#                 elif pred == 0 and label == 1:
#                     FN_count += 1
#                 elif pred == 0 and label == 0:
#                     TN_count += 1

#     # 打印最终统计结果
#     total_pos = TP_count + FN_count
#     total_neg = FP_count + TN_count
    
#     FP_rate = FP_count / total_neg if total_neg > 0 else 0
#     FN_rate = FN_count / total_pos if total_pos > 0 else 0
#     TP_rate = TP_count / total_pos if total_pos > 0 else 0
#     TN_rate = TN_count / total_neg if total_neg > 0 else 0
    
#     # 计算 Precision 和 Recall
#     # Precision: TP / (TP + FP)
#     precision = TP_count / (TP_count + FP_count) if (TP_count + FP_count) > 0 else 0
#     # Recall: TP / (TP + FN)
#     recall = TP_count / (TP_count + FN_count) if (TP_count + FN_count) > 0 else 0

#     print(f"FP rate = {FP_rate:.4f}")
#     print(f"FN rate = {FN_rate:.4f}")
#     print(f"TP rate = {TP_rate:.4f}")
#     print(f"TN rate = {TN_rate:.4f}")
    
#     print(f"Total Positives: {total_pos}")
#     print(f"Total Negatives: {total_neg}")
#     print(f"FP = {FP_count}, FN = {FN_count}, TP = {TP_count}, TN = {TN_count}")
    
#     print("-" * 30)
#     print(f"Final Precision = {precision:.4f}")
#     print(f"Final Recall = {recall:.4f}")
#     print("-" * 30)
#     return total_pos, total_neg 
def test(args):
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    random_seed = 42 
    random.seed(random_seed)

    # 加载测试数据集，这里省略了数据加载部分，保持原样
    dataset_path_pos = '/mnt/mnt1/tyy/data/positive/test_pos.npy'
    train_data_pos = np.load(dataset_path_pos, allow_pickle=True)
    train_data_pos = list(train_data_pos)
    random.shuffle(train_data_pos)
    train_data_pos = train_data_pos[:len(train_data_pos) // 10]
    
    dataset_path_neg = '/mnt/mnt1/tyy/data/newnegative/test_neg.npy'
    train_data_neg = np.load(dataset_path_neg, allow_pickle=True)
    train_data_neg = list(train_data_neg)
    
    test_dataset = create_episode_dataset(train_data_pos, train_data_neg,'total')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 步骤 1: 加载两个模型
    model_stage1 = Reward_Model()
    model_stage1.load_state_dict(torch.load('/home/teamcommon/tyy/MyLander/Env_agent/stage1/40new_rw_model_f5_best.pt'))
    
    from utils.criticality_model import Criticality_model_trans
    model_stage2 = Criticality_model_trans()
    model_stage2.load_state_dict(torch.load('/home/teamcommon/tyy/MyLander/Env_agent/stage3/reward0_stage3_final_model.pt'))

    model_stage1.eval().to(device)
    model_stage2.eval().to(device)
    
    # 定义两个列表来保存所有样本的最终分数和真实标签
    all_final_scores = []
    all_true_labels = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            labels = labels.reshape(-1, 2)
            labels_cls = torch.tensor(labels[:, 0], dtype=torch.int64)
            bs = inputs.shape[0]

            # 步骤 2: 使用 Stage 1 模型进行预测
            outputs_stage1 = model_stage1(inputs)
            rewards_stage1 = outputs_stage1["rewards"][:, -1]
            
            # 拷贝 Stage 1 的预测分数作为初始的最终预测分数
            final_scores = rewards_stage1.cpu().numpy().copy()

            # 步骤 3 & 4: 筛选 Stage 1 预测为正样本的所有数据，交给 Stage 2 重新预测
            stage1_pos_indices = np.where(final_scores > -1)[0]
            
            if len(stage1_pos_indices) > 0:
                stage1_pos_inputs = inputs[stage1_pos_indices]
                outputs_stage2, _, _, _ = model_stage2(stage1_pos_inputs, stage1_pos_inputs, 0.5)
                rewards_stage2 = outputs_stage2[:, -1]
                
                # 步骤 5: 更新 Stage 1 预测分数，用 Stage 2 的结果覆盖
                if len(stage1_pos_indices) == len(rewards_stage2)//2:
                    final_scores[stage1_pos_indices] = rewards_stage2.cpu().numpy()[:len(rewards_stage2)//2]
                else:
                    print(f"Warning: Shape mismatch. Stage1 indices: {len(stage1_pos_indices)}, Stage2 predictions: {len(rewards_stage2)}. Skipping this batch.")
                    continue
            
            # 将当前批次的分数和标签添加到总列表中
            all_final_scores.extend(final_scores)
            all_true_labels.extend(labels_cls.cpu().numpy())

    # 调用绘图函数来绘制最终的分数图
    plot_final_scores(all_final_scores, all_true_labels)
    
    # 统计最终结果
    TP_count, FN_count, FP_count, TN_count = 0, 0, 0, 0
    final_predictions = np.array(all_final_scores) > 0.5 # 假设 Stage 2 的最终阈值为0.5
    all_true_labels = np.array(all_true_labels)

    for k in range(len(all_final_scores)):
        pred = final_predictions[k]
        label = all_true_labels[k].item()
        
        if pred == 1 and label == 1:
            TP_count += 1
        elif pred == 1 and label == 0:
            FP_count += 1
        elif pred == 0 and label == 1:
            FN_count += 1
        elif pred == 0 and label == 0:
            TN_count += 1

    # 打印最终统计结果
    total_pos = TP_count + FN_count
    total_neg = FP_count + TN_count
    
    FP_rate = FP_count / total_neg if total_neg > 0 else 0
    FN_rate = FN_count / total_pos if total_pos > 0 else 0
    TP_rate = TP_count / total_pos if total_pos > 0 else 0
    TN_rate = TN_count / total_neg if total_neg > 0 else 0
    
    precision = TP_count / (TP_count + FP_count) if (TP_count + FP_count) > 0 else 0
    recall = TP_count / (TP_count + FN_count) if (TP_count + FN_count) > 0 else 0

    print(f"FP rate = {FP_rate:.4f}")
    print(f"FN rate = {FN_rate:.4f}")
    print(f"TP rate = {TP_rate:.4f}")
    print(f"TN rate = {TN_rate:.4f}")
    
    print(f"Total Positives: {total_pos}")
    print(f"Total Negatives: {total_neg}")
    print(f"FP = {FP_count}, FN = {FN_count}, TP = {TP_count}, TN = {TN_count}")
    
    print("-" * 30)
    print(f"Final Precision = {precision:.4f}")
    print(f"Final Recall = {recall:.4f}")
    print("-" * 30)


from sklearn.metrics import precision_recall_curve, auc
def picPR(all_scores, all_labels, filename):
    """
    绘制并保存PR曲线。

    Args:
        all_scores (list or np.array): 所有预测分数的列表或数组。
        all_labels (list or np.array): 所有真实标签的列表或数组。
        filename (str): 保存PR曲线图的文件名（包含路径和扩展名）。
    """
    # 确保 all_labels 和 all_scores 的数据类型正确
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # 检查标签中是否同时包含正负样本，这是绘制PR曲线的前提
    if len(np.unique(all_labels)) < 2:
        print("Warning: PR curve cannot be plotted with only one class present in labels.")
        return

    # 计算 PR 曲线
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    average_precision = auc(recall, precision)

    # 绘制曲线
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    # 保存和关闭图像
    plt.savefig(filename)
    plt.close()
    print(f"PR curve saved to {filename}")

def pic(loss_pos, loss_neg):
    markers = ['o', '^']
    markers = ['o', '^']
    colors = ['coral','dodgerblue',]
    labels = ['positive','negative']
    loss = [loss_neg, loss_pos]

    plt.figure(figsize=(14, 5))

    plt.title('Scores')
    plt.ylabel('Scores'); plt.xlabel('Index')
    #plt.subplot(122)
    #plt.subplot(121)

    plt.scatter(range(len(loss_neg)), 
                loss_neg,  
                alpha=0.7, 
                marker=markers[1], 
                c=colors[1], 
                label=labels[1])
    plt.scatter([k * (len(loss_neg)/len(loss_pos)) for k in range(len(loss_pos))], 
            loss_pos,  
            alpha=0.7, 
            marker=markers[0], 
            c=colors[0], 
            label=labels[0])
    

    
    plt.title('Scores')
    plt.ylabel('Scores'); plt.xlabel('Index')
    plt.savefig('/home/teamcommon/tyy/MyLander/Env_agent/stage1/40new_test_reward_model_train_450_f5.png')

def check(reward):
    if (reward >= -2.705765962600708) and (reward <= -0.9136561155319214):
        return 1
    if (reward >= 1.700473666191101) and ( reward <= 2.2):
        return 1
    if (reward >= 3.3319625854492188) and ( reward <= 4.127645492553711):
        return 1
    if (reward >= 7.22815465927124):
        return 1
    else:
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input parameters
    # 2048
    parser.add_argument('--batch_size', default=516, type=int, help='batch size')
    parser.add_argument('--output_model_prefix', default='mlp')

    # Train parameters
    parser.add_argument('--start_epoch', default=0, type=int, help='the number of epochs')
    parser.add_argument('--epochs', default=100, type=int, help='the number of epochs')
    # 1e-4
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--final_ratio',default=0,type=int,help='ratio of negative / positive')
    parser.add_argument('--start_ratio',default=0,type=int, help='ratio of negative / positive')
    parser.add_argument('--warm_up_epochs',default=20,type=int)
    # 0.546
    parser.add_argument('--threshold', default=0.5)

    # Model parameters
    parser.add_argument('--input_dim', default=11, type=int, help='the number of classes')
    parser.add_argument('--embed_dim', default=256, type=int, help='the number of expected features in the mlp')
    parser.add_argument('--embed_dim_1', default=256, type=int, help='the number of expected features in the mlp')
    parser.add_argument('--embed_dim_2', default=1024, type=int, help='the number of expected features in the mlp')
    parser.add_argument('--n_layers', default=6, type=int, help='the number of heads in the multi-head attention network')
    parser.add_argument('--n_heads', default=8, type=int, help='the number of multi-head attention heads')
    parser.add_argument('--dropout', default=0.1, type=float, help='the residual dropout value')
    parser.add_argument('--ffn_dim', default=1024, type=int, help='the dimension of the feedforward network')
    parser.add_argument('--num_classes', default=2, type=int, help='the number of classes')
    parser.add_argument('--max_seq_len', default=11, type=int, help='the number of expected features in the mlp')

    # path
    parser.add_argument('--dataset_dir',default='/mnt1/brx/dataset/')
    parser.add_argument('--log_dir', default='new_log/log')
    parser.add_argument('--is_resume',default=0)
    parser.add_argument('--is_train',default=0)
 
    parser.add_argument('--model_path',default='new_log/model/mlp.ep99')
    parser.add_argument('--best_model_path',default='new_log/model/mlp.ep680')
    parser.add_argument('--first_model_path',default='new_log/model/mlp.ep546')
    parser.add_argument('--two_model',default=1)

    args = parser.parse_args()
    
    if args.is_train:
        main(args)
    else:
        pos, neg = test(args)
        
        #pic(pos, neg)


