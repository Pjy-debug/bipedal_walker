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


'''def test(args):
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    dataset_path_pos = '/mnt/mnt1/tyy/data/positive/test.npy'
    train_data_pos = np.load(dataset_path_pos, allow_pickle=True)
    train_data_pos = list(train_data_pos)
    
    dataset_path_neg = '/mnt/mnt1/tyy/data/newnegative/test.npy'
    train_data_neg = np.load(dataset_path_neg, allow_pickle=True)
    train_data_neg = list(train_data_neg)
    
    test_dataset = create_episode_dataset(train_data_pos, train_data_neg,'total')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    
    model = Reward_Model()
    model.load_state_dict(torch.load('/home/teamcommon/tyy/MyLander/Env_agent/stage1/rw_model_f5_best.pt'))
    
    losses, accs, precisions, recalls, TPRs, FPRs = 0, 0, 0, 0, 0, 0
    cls_losses, sc_losses = 0, 0
    n_batches, n_samples = len(test_loader),len(test_loader.dataset)
    #self.train_neg_iter = iter(self.train_neg_loader)
    model.eval().to(device)
    #model2.eval().to(device)
    
    pos_scores = []
    neg_scores = []
    FN = 0
    FP = 0
    FP_samples = []
    FN_samples = []
    g1 = []
    g2 = []
    g3 = []
    g4 = []
    TP = 0
    TN = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # test data
            inputs, labels = data[0], data[1]
            inputs = inputs.to(device)
            labels = labels.to(device)
            #是否需要改成labels = labels.reshape(-1,2)？
            labels = labels.reshape(-1,2)
            labels_cls = torch.tensor(labels[:,0],dtype=torch.int64)
            labels_dist = labels[:,1]
            bs = inputs.shape[0]

            outputs = model(inputs)

            rewards = outputs["rewards"]
            for k in range(bs):
                cur_input = inputs[k].squeeze(0).clone().detach().cpu().numpy()
                reward = rewards[k,-1].item()
                label = labels_cls[k].item()
                if label == 1:
                    pos_scores.append(reward)
                else:
                    neg_scores.append(reward)
                
                """
                if reward > 2.5:
                    pred = check(reward2)
                else:
                    pred = 0
                
                if pred == 1:
                    if label == 1:
                        TP += 1
                        TP_samples.append(cur_input)
                    else:
                        FP += 1
                        FP_samples.append(cur_input)
                """
                # 2.75 false data
                if reward < -0.05:
                    pred = 0
                else:
                    pred = 1
                if pred == 1 and label==0:
                    FP_samples.append(cur_input)
                    FP += 1
                elif pred == 0 and label == 1:
                    FN_samples.append(cur_input)
                    FN += 1
                elif pred == 0 and label == 0:
                    TN += 1
                elif pred == 1 and label == 1:
                    TP += 1
    print (f"neg len = f{len(neg_scores)}")
    print (f"pos len = f{len(pos_scores)}")
    FP_rate = FP / len(neg_scores)
    print(f"FP rate = {FP_rate}")
    FN_rate = FN / len (pos_scores)
    print(f"FN rate = {FN_rate}")
    TP_rate = TP / len(pos_scores)
    TN_rate = TN / len(neg_scores)
    print(f"tp rate = {TP_rate}")
    print(f"tn rate = {TN_rate}")
                
                
    """
    print(min(g1),max(g1))
    print(min(g2),max(g2))
    print(min(g3),max(g3))
    print(min(g4),max(g4))

    print(FP, FN)
    np.save(f'/home/teamcommon/tyy/MyLander/Env_agent/stage1/test_FP_final5.npy', FP_samples, allow_pickle=True)
    #np.save(f'/root/autodl-tmp/data/distill_dataset/rm_TP_2.npy', TP_samples, allow_pickle=True)
    
    #np.save(f'model/pos_scores_train_450.npy', pos_scores, allow_pickle=True)
    #np.save(f'model/neg_scores_train_450.npy', neg_scores, allow_pickle=True)

    return pos_scores, neg_scores'''

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
    
def test(args):
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # 设置一个固定的随机种子，确保可复现性
    random_seed = 42  
    random.seed(random_seed)

    dataset_path_pos = '/mnt/mnt1/tyy/data/positive/test_pos.npy'
    train_data_pos = np.load(dataset_path_pos, allow_pickle=True)
    train_data_pos = list(train_data_pos)
    #修改
    random.shuffle(train_data_pos)  # 随机打乱正样本数据
    train_data_pos = train_data_pos[:len(train_data_pos) // 10]
    
    dataset_path_neg = '/mnt/mnt1/tyy/data/newnegative/test_neg.npy'
    train_data_neg = np.load(dataset_path_neg, allow_pickle=True)
    train_data_neg = list(train_data_neg)
    
    test_dataset = create_episode_dataset(train_data_pos, train_data_neg,'total')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    #修改加在的模型
    model_stage1 = Reward_Model()
    model_stage1.load_state_dict(torch.load('/home/teamcommon/tyy/MyLander/Env_agent/stage1/40new_rw_model_f5_best.pt'))
    #修改
    from utils.criticality_model import Criticality_model_trans
    model_stage2 = Criticality_model_trans()
    model_stage2.load_state_dict(torch.load('/home/teamcommon/tyy/MyLander/Env_agent/stage2/model/trans_49.pt')) 
    losses, accs, precisions, recalls, TPRs, FPRs = 0, 0, 0, 0, 0, 0
    cls_losses, sc_losses = 0, 0
    n_batches, n_samples = len(test_loader),len(test_loader.dataset)
    #self.train_neg_iter = iter(self.train_neg_loader)
    model_stage1.eval().to(device)
    model_stage2.eval().to(device)
    #model2.eval().to(device)
    
    pos_scores = []
    neg_scores = []
    FN = 0
    FP = 0
    FP_samples = []
    FN_samples = []
    all_scores = []
    all_labels = []
    # 初始化统计变量
    FP_count, FN_count, TP_count, TN_count = 0, 0, 0, 0
    FP_samples = []
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # test data
            inputs, labels = data[0], data[1]
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.reshape(-1,2)
            labels_cls = torch.tensor(labels[:,0],dtype=torch.int64)
            labels_dist = labels[:,1]
            bs = inputs.shape[0]

            outputs = model_stage1(inputs) 
            rewards_stage1 = outputs["rewards"]
            #修改
            # 初始化一个列表来存放最终的分数
            final_rewards = []
            for k in range(bs):
                # 检查 Stage 1 模型的预测分数是否高于阈值
                if rewards_stage1[k, -1].item() > 0.3:
                    # 如果高于阈值，用 Stage 3 模型进行二次预测
                    #outputs_stage3 = model_stage3(inputs[k:k+1])
                    #reward = outputs_stage2["rewards"][:, -1].item()
                    reward = rewards_stage1[k, -1].item()
                    final_rewards.append(reward)
                else:
                    # 如果低于阈值，直接用 Stage 1 的分数
                    reward = rewards_stage1[k, -1].item()
                    #reward = -0.05
                    final_rewards.append(reward)
            
            # 将最终的分数列表转换为张量
            final_rewards_tensor = torch.tensor(final_rewards).to(device)

            all_scores.extend(final_rewards_tensor.clone().cpu().numpy().tolist())
            all_labels.extend(labels_cls.clone().cpu().numpy().tolist())

            # all_scores.extend(rewards[:, -1].clone().cpu().numpy().tolist())
            # all_labels.extend(labels_cls.clone().cpu().numpy().tolist())
            for k in range(bs):
                cur_input = inputs[k].squeeze(0).clone().detach().cpu().numpy()
                #reward = rewards[k,-1].item()
                #修改
                reward = final_rewards_tensor[k].item() 
                label = labels_cls[k].item()
                if label == 1:
                    pos_scores.append(reward)
                else:
                    neg_scores.append(reward)
                
                #把threshold新改成0.2
                if reward < 0.3:
                    pred = 0
                else:
                    pred = 1
                if pred == 1 and label==0:
                    FP_samples.append(cur_input)
                    FP += 1
                elif pred == 0 and label == 1:
                    FN_samples.append(cur_input)
                    FN += 1
    FP_rate = FP / len(neg_scores)
    print(f"FP rate = {FP_rate}")
    FN_rate = FN / len (pos_scores)
    print(f"FN rate = {FN_rate}")
    print(f"FP = {FP}")
    print(f"FN = {FN}")
    print(f"TP = {len(pos_scores)-FN}")
    print(f"TN = {len(neg_scores)-FP}")
    #print(f"recall = {(len(pos_scores)-FN)/len(pos_scores)}")
    #print(f"precision = {(len(pos_scores)-FN)/(len(pos_scores)-FN+FP)}")
    #picPR(all_scores=all_scores, all_labels=all_labels, filename="/home/teamcommon/tyy/MyLander/Env_agent/stage1/stage3_40new_PR_curve.png")


    np.save(f'/home/teamcommon/tyy/MyLander/Env_agent/stage1/40new_val_FP_final5.npy', FP_samples, allow_pickle=True)
    #evaluate_with_thresholds(all_scores, all_labels, min_threshold=-0.1, max_threshold=0.4)
    #np.save(f'/root/autodl-tmp/data/distill_dataset/rm_TP_2.npy', TP_samples, allow_pickle=True)
    
    #np.save(f'model/pos_scores_train_450.npy', pos_scores, allow_pickle=True)
    #np.save(f'model/neg_scores_train_450.npy', neg_scores, allow_pickle=True)

    return pos_scores, neg_scores

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
        pic(pos, neg)


