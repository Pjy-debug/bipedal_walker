import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from criticality_model import Reward_Model
from .criticality_model import Reward_Model
from pathlib import Path
import numpy as np
import math
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import argparse
import  matplotlib.pyplot as plt

class Trainer:
    def __init__(self, args, train_pos_loader=None,train_neg_loader=None, test_pos_loader=None, test_neg_loader= None):
        self.args = args
        self.train_pos_loader = train_pos_loader
        self.train_neg_loader = train_neg_loader
        self.train_neg_iter = iter(self.train_neg_loader)
        self.test_neg_loader = test_neg_loader
        #self.test_neg_iter = iter(self.test_neg_loader)
        self.test_pos_loader = test_pos_loader
        self.device = 'cuda:1' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

        if not args.is_resume:
            self.model = Reward_Model()
            
        # resume = 1 恢复  
        else:
            self.model = torch.load(args.model_path)
            """
            for name,param in self.model.named_parameters():
                if 'cls_head' not in name:
                    param.requires_grad=False
            """
            
        self.model.to(self.device)
        self.L = args.epochs - args.start_epoch
        self.epsilon = 0.5

        self.optimizer = optim.Adam(self.model.parameters(), args.lr)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-8)
        self.lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=2, eta_min=1e-5)
    

    def train(self, epoch):
        print("train start")
        losses, accs, precisions, recalls, TPRs, FPRs = 0, 0, 0, 0, 0, 0
        cls_losses, sc_losses = 0, 0
        n_batches, n_samples = len(self.train_pos_loader),len(self.train_pos_loader.dataset)*2
        #self.train_neg_iter = iter(self.train_neg_loader)
        self.model.train()
        correct_predictions = 0
        total_predictions = 0
        scores = 0

        # --- 新增: 用于存储所有训练样本分数的列表 ---
        all_chosen_scores = []
        all_rejected_scores = []

        for i, data1 in enumerate(self.train_pos_loader):
            try:
                data2 = next(self.train_neg_iter)
            except StopIteration:
                self.train_neg_iter = iter(self.train_neg_loader)
                data2 = next(self.train_neg_iter)
            # pos data
            input1, label1 = data1[0],data1[1]
            # neg data
            input2, label2 = data2[0],data2[1]
            '''
            inputs = torch.cat((input1,input2),dim=0)
            labels = torch.cat((label1,label2),dim=0)
            '''
            # 修改: 不再拼接输入，分别送入模型
            chosen_as = input1.to(self.device)
            rejected_as = input2.to(self.device)

            # 新增: 分别调用模型并获取rewards
            chosen_outputs = self.model(chosen_as)
            rejected_outputs = self.model(rejected_as)
            chosen_rewards = chosen_outputs["rewards"]
            rejected_rewards = rejected_outputs["rewards"]
            # 新增: 从rewards中提取最后一帧的奖励分数进行对比
            chosen_mean_scores = chosen_rewards[:, -1]
            rejected_mean_scores = rejected_rewards[:, -1]

            # 修改: 在 Trainer 中计算损失
            margin = 0.5
            loss = torch.relu(rejected_mean_scores - chosen_mean_scores + margin).mean()
            

            #inputs = inputs.to(self.device)
            #labels = labels.to(self.device)
            #是否要改成labels = labels.reshape(-1,2)?
            #labels = labels.reshape(-1,2)
            '''
            labels_cls = torch.tensor(labels[:,0],dtype=torch.int64)
            labels_dist = labels[:,1]
            #print(f"len of inputs{len(inputs)}")
            outputs = self.model(inputs)    
            loss = outputs['loss']
            
            chosen = outputs["chosen_mean_scores"]
            rejected = outputs["rejected_mean_scores"]
            '''
            # --- 新增: 将每个批次的分数追加到列表中 ---
            all_chosen_scores.extend(chosen_mean_scores.cpu().detach().numpy())
            all_rejected_scores.extend(rejected_mean_scores.cpu().detach().numpy())

            # 新的统计方式：chosen > -0.2 的数量 + rejected < -0.2 的数量
            correct_chosen_count = (chosen_mean_scores > -0.05).sum().item()
            correct_rejected_count = (rejected_mean_scores < -0.05).sum().item()
            correct_predictions += (correct_chosen_count + correct_rejected_count)
            #correct_predictions += (chosen > rejected).sum()
            total_predictions += (chosen_mean_scores.shape[0] + rejected_mean_scores.shape[0])

            """累加每个step的平均chosen分值"""
            #scores += outputs["chosen_mean_scores"].mean().float()
            scores += chosen_mean_scores.mean().float()

            losses += loss.item()
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            if i % 1000 == 0:
                print('Train Epoch {} Iteration {} ({}/{})\tLoss: {:.4f} / Acc: {:.4f}'.format(epoch,i, i, n_batches, losses / (i+1), correct_predictions / total_predictions))
        
        acc = correct_predictions / total_predictions
        scores = scores / n_batches
    
        print('Train dataset: Epoch : {}\t>\tLoss: {:.4f} / Acc: {:.4f} / Scores:{:.4f}'.format(epoch, losses / n_batches, acc, scores))

        # --- 修改: 构造文件名并调用 pic 函数 ---
        # 构建你想要的特定文件名，并在文件名中包含 epoch
        filename = f'/home/teamcommon/tyy/MyLander/Env_agent/stage1/picture/test_reward_model_train_epoch_{epoch}_f5.png'
        # 调用 pic 函数保存图片
        pic(all_chosen_scores, all_rejected_scores, filename)

        return scores, acc

    def validate(self, epoch):
        print("validate start")
        losses, accs, precisions, recalls, TPRs, FPRs = 0, 0, 0, 0, 0, 0
        cls_losses, sc_losses = 0, 0
        n_batches, n_samples = len(self.test_pos_loader),len(self.test_pos_loader.dataset)*2
        #self.train_neg_iter = iter(self.train_neg_loader)
        self.test_neg_iter = iter(self.test_neg_loader)
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        scores = 0

        # 新增: 收集所有分数和标签以便于绘制PR曲线
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for i, data1 in enumerate(self.test_pos_loader):
                try:
                    data2 = next(self.test_neg_iter)
                except StopIteration:
                    self.test_neg_iter = iter(self.test_neg_loader)
                    data2 = next(self.test_neg_iter)

                # pos data
                input1, label1 = data1[0],data1[1]
                # neg data
                input2, label2 = data2[0],data2[1]

                # 修改: 不再拼接输入，分别送入模型
                chosen_as = input1.to(self.device)
                rejected_as = input2.to(self.device)

                # 新增: 分别调用模型并获取rewards
                chosen_outputs = self.model(chosen_as)
                rejected_outputs = self.model(rejected_as)
                chosen_rewards = chosen_outputs["rewards"]
                rejected_rewards = rejected_outputs["rewards"]
                
                # 新增: 从rewards中提取最后一帧的奖励分数进行对比
                chosen_mean_scores = chosen_rewards[:, -1]
                rejected_mean_scores = rejected_rewards[:, -1]

                # --- 核心修改: 确保两个批次大小一致 ---
                # 获取两个批次中较小的那个大小
                batch_size = min(chosen_mean_scores.size(0), rejected_mean_scores.size(0))
                
                # 对两个张量进行切片，使其批次大小相同
                chosen_mean_scores = chosen_mean_scores[:batch_size]
                rejected_mean_scores = rejected_mean_scores[:batch_size]

                # 修改: 在 Trainer 中计算损失
                margin = 0.5
                loss = torch.relu(rejected_mean_scores - chosen_mean_scores + margin).mean()

                # 新增: 收集所有分数和标签
                all_scores.extend(chosen_mean_scores.cpu().numpy())
                all_scores.extend(rejected_mean_scores.cpu().numpy())
                all_labels.extend([1] * chosen_mean_scores.shape[0])
                all_labels.extend([0] * rejected_mean_scores.shape[0])

                # 新的统计方式：chosen > -0.2 的数量 + rejected < -0.2 的数量
                correct_chosen_count = (chosen_mean_scores > -0.05).sum().item()
                correct_rejected_count = (rejected_mean_scores < -0.05).sum().item()
                correct_predictions += (correct_chosen_count + correct_rejected_count)
                #correct_predictions += (chosen > rejected).sum()
                total_predictions += (chosen_mean_scores.shape[0] + rejected_mean_scores.shape[0])

                """累加每个step的平均chosen分值"""
                scores += chosen_mean_scores.mean().float()

                losses += loss.item()

                if i % 1000 == 0:
                    print('Val Epoch {} Iteration {} ({}/{})\tLoss: {:.4f} / Acc: {:.4f}'.format(epoch,i, i, n_batches, losses / (i+1), correct_predictions / total_predictions))

            acc = correct_predictions / total_predictions
            scores = scores / n_batches

            print('Val dataset: Epoch : {}\t>\tLoss: {:.4f} / Acc: {:.4f} / Scores:{:.4f}'.format(epoch, losses / n_batches, acc, scores))
            # --- 调用 picPR 函数 ---
            # 构造文件名
            #filename = f'/home/teamcommon/tyy/MyLander/Env_agent/stage1/PRcurve/pr_curve_epoch_{epoch}.png'
            # 调用 picPR 函数
            #picPR(all_scores, all_labels, filename)
            #新增
            all_scores = np.array(all_scores)
            all_labels = np.array(all_labels)

            # Set the threshold
            threshold = -0.05
            
            # Make predictions based on the threshold
            y_pred = (all_scores >= threshold).astype(int)

            # Calculate True Positives, False Positives, and False Negatives
            tp = np.sum((y_pred == 1) & (all_labels == 1))
            fp = np.sum((y_pred == 1) & (all_labels == 0))
            fn = np.sum((y_pred == 0) & (all_labels == 1))
            
            # Calculate Precision and Recall
            precision_at_threshold = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_at_threshold = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Print the results
            print(f"\nResults for threshold = {threshold:.2f}:")
            print(f"  Precision: {precision_at_threshold:.4f}")
            print(f"  Recall:    {recall_at_threshold:.4f}")
            # --- End of new code ---
        return scores, acc

    '''def save(self, epoch, model_prefix='rw_model', root='model'):
        path = Path(root) / (model_prefix + '.ep%d' % epoch)
        if not path.parent.exists():
            path.parent.mkdir()
        torch.save(self.model.state_dict(), f'/mnt1/brx/Rocketdata/model/stage1/rw_model_f5_{epoch}.pt')'''
    

    def save(self, epoch, model_prefix='rw_model', root='model', filename=None):
        # ⬅️ 修改 1: 检查是否提供了自定义文件名
        if filename is None:
            # 如果没有提供，使用默认的文件名格式
            save_path = Path(root) / (model_prefix + '.ep%d' % epoch)
        else:
            # 如果提供了，使用自定义的文件名
            save_path = Path(root) / filename
            
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
        # ⬅️ 修改 2: 将保存路径统一为 save_path
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def save_best(self, model_prefix='rw_model', root='model'):
        path = Path(root) / (model_prefix + '.ep')
        if not path.parent.exists():
            path.parent.mkdir()
        torch.save(self.model, path)



import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
import numpy as np

def pic(loss_pos, loss_neg, filename):
    markers = ['o', '^']
    colors = ['coral','dodgerblue',]
    labels = ['positive','negative']

    plt.figure(figsize=(14, 5))

    plt.title('Scores')
    plt.ylabel('Scores'); plt.xlabel('Index')

    # 将正负样本分数合并成一个 NumPy 数组
    all_scores = np.concatenate([loss_pos, loss_neg])

    # 如果 all_scores 为空，则不设置 ylim
    if len(all_scores) > 0:
        # 找到所有分数的最小值和最大值
        min_score = np.min(all_scores)
        max_score = np.max(all_scores)
        
        # 动态设置 y 轴的显示范围
        # 为了有更好的视觉效果，可以留一些空白
        margin = (max_score - min_score) * 0.1  # 留 10% 的空白
        plt.ylim(min_score - margin, max_score + margin)
    # 绘制负样本分数
    plt.scatter(range(len(loss_neg)),
                loss_neg,
                alpha=0.7,
                marker=markers[1],
                c=colors[1],
                label=labels[1])    


    # 绘制正样本分数
    plt.scatter(range(len(loss_pos)),
                loss_pos,
                alpha=0.7,
                marker=markers[0],
                c=colors[0],
                label=labels[0])


    plt.legend()
    # 使用传入的 filename 参数保存图片
    plt.savefig(filename)
    plt.close() # 关闭图形以释放内存


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