'''*************************************************************************
[file name]                 reward_model.py
[description]              奖励模型的训练和验证
[developer]                brx, 2024
[changelog]                修改依赖文件路径导入
[usage]                    
*************************************************************************'''
import sys
import os
# 获取当前文件的绝对路径
current_path = os.path.abspath(__file__)
# 向上一级目录
parent_dir = os.path.dirname(os.path.dirname(current_path))
# 将criticality目录添加到Python的模块搜索路径
sys.path.append(parent_dir)

# 现在可以导入criticality_目录下的函数了
from criticality_.criticality_model import Mlp, Reward_Model, Criticality_model
from plot.draw_two import draw_roc, draw_pr

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import math
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from data_.data_utils import create_dataset,create_pos_dataset,create_neg_dataset, create_distill_dataset,create_dataset_new, create_neg_dataset, create_dataset_test

import argparse
import  matplotlib.pyplot as plt

import tqdm

from statistics import median

class Trainer:
    def __init__(self, args, train_pos_loader=None,train_neg_loader=None,val_pos_loader=None,val_neg_loader=None, test_loader=None):
        self.args = args
        self.train_pos_loader = train_pos_loader
        self.train_neg_loader = train_neg_loader
        self.train_neg_iter = iter(self.train_neg_loader)
        self.val_pos_loader = val_pos_loader
        self.val_neg_loader = val_neg_loader
        self.val_neg_iter = iter(self.val_neg_loader)
        self.test_loader = test_loader
        self.test_iter = iter(self.test_loader)
        self.device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

        if not args.is_resume:
            self.model = Reward_Model()
            
        # resume = 1 恢复  
        else:
            self.model = Reward_Model()
            self.model.load_state_dict(torch.load('/home/teamcommon/pjy/Bipedal_walker/criticality/stage1/storage/rw_149.pt'))
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
        losses, accs, precisions, recalls, TPRs, FPRs = 0, 0, 0, 0, 0, 0
        cls_losses, sc_losses = 0, 0
        n_batches, n_samples = len(self.train_pos_loader),len(self.train_pos_loader.dataset)*2
        self.model.train()
        correct_predictions = 0
        total_predictions = 0
        scores = 0
        
        # for debug
        print(f'training, n_batches is {n_batches}, n_samples is {n_samples}')

        # data1, data2分别为正负样本
        for i, data1 in enumerate(self.train_pos_loader):
            try:
                data2 = next(self.train_neg_iter)
            # 负样本进行了重复遍历（实际没有）
            except StopIteration:
                self.train_neg_iter = iter(self.train_neg_loader)
                data2 = next(self.train_neg_iter)
            
            # pos data
            input1, label1 = data1[0],data1[1]
            # neg data
            input2, label2 = data2[0],data2[1]
            inputs = torch.cat((input1,input2),dim=0)
            labels = torch.cat((label1,label2),dim=0)
            
            # for debug
            # print(label1[:,0].sum(), label2[:,0].sum())
            # for debug
            # print(f'training, inputs shape is {inputs.shape}')

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            labels.reshape(-1,2)
            labels_cls = torch.tensor(labels[:,0],dtype=torch.int64)
            labels_dist = labels[:,1]

            outputs = self.model(inputs)
            
            # for debug
            # print(f"training, outputs['rewards'] shape is {outputs['rewards'].shape}")

            # Strange here, Reward model do not have a return called loss
            loss = outputs['loss']
            
            chosen = outputs["chosen_mean_scores"]
            rejected = outputs["rejected_mean_scores"]
            correct_predictions += (chosen > rejected).sum()
            total_predictions += chosen.shape[0]
            
            # for debug
            # print('chosen shape: ', chosen.shape, 'rejected shape: ', rejected.shape)
            
            """累加每个step的平均chosen分值"""
            scores += outputs["chosen_mean_scores"].mean().float()

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
        return scores, acc

    def validate(self, epoch):
        losses, accs, precisions, recalls, TPRs, FPRs = 0, 0, 0, 0, 0, 0
        cls_losses, sc_losses = 0, 0
        n_batches, n_samples = len(self.val_pos_loader),len(self.val_pos_loader.dataset)*2
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        scores = 0
        
        with torch.no_grad():
            for i, data1 in enumerate(self.val_pos_loader):
                try:
                    data2 = next(self.val_neg_iter)
                except StopIteration:
                    self.val_neg_iter = iter(self.val_neg_loader)
                    data2 = next(self.val_neg_iter)

                # pos data
                input1, label1 = data1[0],data1[1]
                # neg data
                input2, label2 = data2[0],data2[1]
                inputs = torch.cat((input1,input2),dim=0)
                labels = torch.cat((label1,label2),dim=0)
                

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                labels.reshape(-1,2)
                labels_cls = torch.tensor(labels[:,0],dtype=torch.int64)
                labels_dist = labels[:,1]

                outputs = self.model(inputs)

                loss = outputs['loss']

                chosen = outputs["chosen_mean_scores"]
                rejected = outputs["rejected_mean_scores"]
                correct_predictions += (chosen - 1.5 > rejected).sum()
                total_predictions += chosen.shape[0]

                """累加每个step的平均chosen分值"""
                scores += outputs["chosen_mean_scores"].mean().float()

                losses += loss.item()

                if i % 1000 == 0:
                    print('Val Epoch {} Iteration {} ({}/{})\tLoss: {:.4f} / Acc: {:.4f}'.format(epoch,i, i, n_batches, losses / (i+1), correct_predictions / total_predictions))

            acc = correct_predictions / total_predictions
            scores = scores / n_batches

            print('Val dataset: Epoch : {}\t>\tLoss: {:.4f} / Acc: {:.4f} / Scores:{:.4f}'.format(epoch, losses / n_batches, acc, scores))
        return scores, acc

    def save(self, epoch, model_prefix='model', root='new_log/reward_model'):
        """
        path = Path(root) / (model_prefix + '.ep%d' % epoch)
        if not path.parent.exists():
            path.parent.mkdir()
        """
        torch.save(self.model.state_dict(), f'/home/teamcommon/pjy/Bipedal_walker/criticality/stage1/storage/rw_{epoch}.pt')

    def save_best(self, model_prefix='best_model', root='/home/teamcommon/pjy/Bipedal_walker/criticality/stage1/storage'):
        path = Path(root) / (model_prefix + '.pt')
        if not path.parent.exists():
            path.parent.mkdir()
        torch.save(self.model.state_dict(),  path)

    def test_new(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        losses, accs, precisions, recalls, TPRs, FPRs = 0, 0, 0, 0, 0, 0
        cls_losses, sc_losses = 0, 0
        n_batches = len(self.test_loader)

        model = Reward_Model()
        # load the last mmodel
        print('loading model...')
        model.load_state_dict(torch.load('/home/teamcommon/pjy/Bipedal_walker/criticality/stage1/storage/rw_149.pt'))
        model.eval().to(device)

        correct_predictions = 0
        total_predictions = 0
        scores = 0

        pos_scores=[]
        neg_scores=[]
        
        TP, FP, TN, FN = 0, 0, 0, 0

        # for debug
        
        TPR_list=[]
        FPR_list=[]
        precison_list=[]
        turns=19
        for test_turn in tqdm.tqdm(range(turns)):
            reward_threshold=-0.999+0.111*test_turn

            for i, data in tqdm.tqdm(enumerate(self.test_loader)):
                input, label = data[0],data[1]
                inputs = input
                labels = label

                # for debug
                # print(label[:,0].sum())
                # for debug
                # print(f'training, inputs shape is {inputs.shape}')
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels.reshape(-1,2)
                labels_cls = torch.tensor(labels[:,0],dtype=torch.int64).clone().detach()
                labels_dist = labels[:,1]
                bs = inputs.shape[0]
                outputs = model(inputs)

                rewards = outputs["rewards"]

                # for debug
                # print(label)
                # print(labels_cls)

                max_reward =rewards[:, -1].max()
                min_reward =rewards[:, -1].min()
                mid_reward = (max_reward + min_reward) / 2
                med_reward =median(rewards[:,-1])
                reward_threshold_new = mid_reward + reward_threshold * (max_reward - min_reward) * 0.5
                # print(f'Reward threshold new is {reward_threshold_new}')
                for k in range(bs):
                    cur_input = inputs[k].squeeze(0).clone().detach().cpu().numpy()
                    reward = rewards[k,-1].item()
                    #reward2 = rewards2[k,-1].item()
                    label = labels_cls[k].item()

                    if reward > reward_threshold_new:
                        if label==1:
                            TP+=1
                        else:
                            FP+=1
                    else:
                        if label==0:
                            TN+=1
                        else:
                            FN+=1

                    if test_turn==turns-1:    
                        if label == 1:
                            pos_scores.append(reward)
                        else:
                            neg_scores.append(reward)
            
            print(f'reward_threshold is:{reward_threshold}')
            print(f'TP is {TP}, FP is {FP}, TN is {TN}, FN is {FN}')
            TPR=TP/(TP+FN) # recall rate
            FPR=FP/(FP+TN)
            precison=TP/(TP+FP)
            print(f'TPR is {TPR}, FPR is {FPR}, precison is {precison}')

            TPR_list.append(TPR)
            FPR_list.append(FPR)
            precison_list.append(precison)

            TP, FP, TN, FN = 0, 0, 0, 0


        print(f'TPR_list is {TPR_list}, FPR_list is {FPR_list}, precison_list is {precison_list}')
        return pos_scores, neg_scores, TPR_list, FPR_list, precison_list

    def test_new_2(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        losses, accs, precisions, recalls, TPRs, FPRs = 0, 0, 0, 0, 0, 0
        cls_losses, sc_losses = 0, 0
        n_batches = len(self.test_loader)

        model = Reward_Model()
        # load the last mmodel
        print('loading model...')
        model.load_state_dict(torch.load('/home/teamcommon/pjy/Bipedal_walker/criticality/stage1/storage/rw_149.pt'))
        model.eval().to(device)

        correct_predictions = 0
        total_predictions = 0
        scores = 0

        pos_scores=[]
        neg_scores=[]
        
        turns=19

        TP=[0 for _ in range(turns)]
        FP=[0 for _ in range(turns)]
        TN=[0 for _ in range(turns)]
        FN=[0 for _ in range(turns)]

        for i, data in tqdm.tqdm(enumerate(self.test_loader)):
            input, label = data[0],data[1]
            inputs = input
            labels = label
            # for debug
            # print(label[:,0].sum())
            # for debug
            # print(f'training, inputs shape is {inputs.shape}')
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels.reshape(-1,2)
            labels_cls = torch.tensor(labels[:,0],dtype=torch.int64).clone().detach()
            labels_dist = labels[:,1]
            bs = inputs.shape[0]
            outputs = model(inputs)
            rewards = outputs["rewards"]
            # for debug
            # print(label)
            # print(labels_cls)
            
            max_reward =rewards[:, -1].max()
            min_reward =rewards[:, -1].min()
            mid_reward = (max_reward + min_reward) / 2
            med_reward =median(rewards[:,-1])

            for k in range(bs):
                cur_input = inputs[k].squeeze(0).clone().detach().cpu().numpy()
                reward = rewards[k,-1].item()
                #reward2 = rewards2[k,-1].item()
                label = labels_cls[k].item()

                if label == 1:
                    pos_scores.append(reward)
                else:
                    neg_scores.append(reward)
                
                for test_turn in range(turns):
                    reward_threshold=-0.999+0.111*test_turn
                    reward_threshold_new = mid_reward + reward_threshold * (max_reward - min_reward) * 0.5
                    # print(f'Reward threshold new is {reward_threshold_new}')
                    if reward > reward_threshold_new:
                        if label==1:
                            TP[test_turn]+=1
                        else:
                            FP[test_turn]+=1
                    else:
                        if label==0:
                            TN[test_turn]+=1
                        else:
                            FN[test_turn]+=1
                    
        
        print(f'reward_threshold is:{reward_threshold}')
        print(f'TP is {TP}\nFP is {FP}\nTN is {TN}\nFN is {FN}\n')
        TPR = [tp / (tp + fn) for tp, fn in zip(TP, FN)] # recall rate
        FPR = [fp / (fp + tn) for fp, tn in zip(FP, TN)]
        precison = [tp / (tp + fp) for tp, fp in zip(TP, FP)]
        print(f'TPR is {TPR}\nFPR is {FPR}\nprecison is {precison}\n')
        
        return pos_scores, neg_scores, TPR, FPR, precison

def data_load_and_divide_712():
    dataset_path_pos = '/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data_pos_new/transitions_all.npy'
    print('pos data is being loaded...')
    data_pos = np.load(dataset_path_pos, allow_pickle=True)

    # for all neg data and neg val data
    print('neg data is being loaded...')
    data_neg=[]
    dataset_path_neg_root = '/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data_neg_new/'
    file_names = os.listdir(dataset_path_neg_root)
    neg_data_list = [
        dataset_path_neg_root + file_name for file_name in file_names if (file_name.endswith('.npy') and not file_name.endswith('all.npy'))]
    for item in tqdm.tqdm(neg_data_list):
        data_neg+=np.load(item, allow_pickle=True).tolist()

    # print(f'len of data pos is {len(data_pos)}, len of data neg is {len(data_neg)}')
    # len of data pos is 587, len of data neg is 14991542   
    train_pos_size= round(len(data_pos)*0.7)
    val_pos_size  = round(len(data_pos)*0.1)
    test_pos_size = round(len(data_pos)*0.2)
    train_neg_size= round(len(data_neg)*0.7)
    val_neg_size  = round(len(data_neg)*0.1)
    test_neg_size = round(len(data_neg)*0.2)
    print(f'train_pos_size is {train_pos_size}, val_pos_size is {val_pos_size}, test_pos_size is {test_pos_size}')
    print(f'train_neg_size is {train_neg_size}, val_neg_size is {val_neg_size}, test_neg_size is {test_neg_size}')
    data_pos_train=data_pos[0:train_pos_size]
    data_pos_val=data_pos[train_pos_size:train_pos_size+val_pos_size]
    data_pos_test=data_pos[train_pos_size+val_pos_size:train_pos_size+val_pos_size+test_pos_size]
    data_neg_train=data_neg[0:train_neg_size]
    data_neg_val=data_neg[train_neg_size:train_neg_size+val_neg_size]
    data_neg_test=data_neg[train_neg_size+val_neg_size:train_neg_size+val_neg_size+test_neg_size]
    return [data_pos_train, data_pos_val, data_pos_test, data_neg_train, data_neg_val, data_neg_test]

def main_new(args):
    print(args)
    writer = SummaryWriter(log_dir=args.log_dir)
    train_info = []
    val_info = []
    test_info = []
    
    # divide data
    print('Divide data')
    [train_data_pos, val_data_pos, test_data_pos, train_data_neg, val_data_neg, test_data_neg] = data_load_and_divide_712()
    
    # build dataset
    print('Build dataset')
    train_dataset_pos = create_pos_dataset(train_data_pos)
    val_dataset_pos = create_pos_dataset(val_data_pos)
    train_dataset_neg = create_neg_dataset(train_data_neg)
    val_dataset_neg = create_neg_dataset(val_data_neg)
    test_dataset_pos = create_pos_dataset(test_data_pos)
    test_dataset_neg = create_neg_dataset(test_data_neg)

    # Build DataLoader
    print('Build DataLoader')
    # too big batch size will cause loader failing to load data
    train_pos_loader = DataLoader(train_dataset_pos,batch_size=128,shuffle=True,drop_last = True)
    train_neg_loader = DataLoader(train_dataset_neg,batch_size=128,shuffle=True,drop_last = True)
    
    val_pos_loader = DataLoader(val_dataset_pos, batch_size=32, shuffle=True,drop_last = True)
    val_neg_loader = DataLoader(val_dataset_neg, batch_size=32, shuffle=True,drop_last = True)
    
    test_dataset=test_dataset_pos+test_dataset_neg
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True ,drop_last = True)

    # Build Trainer
    trainer = Trainer(args,train_pos_loader,train_neg_loader,val_pos_loader,val_neg_loader, test_loader)
    # Train & Validate
    print('Train & Validate')
    old_data_idx = 0
    val_acc_best=0
    best_model_index=0
    for epoch in tqdm.tqdm(range(args.start_epoch,args.epochs)):        
        print('Train Epoch: {}'.format(epoch))
        train_neg_loader = DataLoader(train_dataset_neg,batch_size=16,shuffle=True,drop_last = True)
        trainer.train_neg_loader = train_neg_loader
        trainer.train_neg_iter = iter(train_neg_loader)
        train_scores, train_acc = trainer.train(epoch)
        val_scores, val_acc = trainer.validate(epoch)
        if val_acc>val_acc_best:
            val_acc_best = val_acc
            best_model_index = epoch
        trainer.save(epoch, args.output_model_prefix)
    trainer.save_best()

    # Test
    print('Test')
    
    test_pos_res, test_neg_res, TPR_list, FPR_list, precison_list = trainer.test_new_2()

    pic(test_pos_res, test_neg_res)
    plt.clf()
    draw_roc(FPR_list, TPR_list, 'statics/roc_curve.png')
    plt.clf()
    draw_pr(TPR_list, precison_list, 'statics/pr_curve.png')
    plt.clf()
    writer.close()
    
def main(args):
    print(args)
    writer = SummaryWriter(log_dir=args.log_dir)
    train_info = []
    val_info = []
    test_info = []

    # build dataset
    #dataset_path_pos = '/root/autodl-tmp/data/dataset/train_dataset_pos_narrow.npy' #old
    # dataset_path_pos = '/root/autodl-tmp/new_nde/train_pos_dataset.npy'
    dataset_path_pos = '/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data_pos/transitions_all.npy'
    train_data_pos = np.load(dataset_path_pos, allow_pickle=True)
    train_data_pos = list(train_data_pos)
    # val_data_pos = list(np.load( '/root/autodl-tmp/new_nde/test_pos_dataset.npy', allow_pickle=True))
    val_data_pos = list(np.load(dataset_path_pos, allow_pickle=True))
    train_dataset_pos = create_pos_dataset(train_data_pos)
    val_dataset_pos = create_pos_dataset(val_data_pos)
    
    #dataset_path_neg = '/root/autodl-tmp/data/distill_dataset/AE_mlp_FP_1.npy'
    #dataset_path_neg = '/root/autodl-tmp/data/distill_dataset/rm_FP.npy' # old
    # dataset_path_neg = '/root/autodl-tmp/new_nde/train_neg_dataset.npy'

    # for simple neg data
    dataset_path_neg = '/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data_neg/transitions_0.npy' #原来这里误写成了transitions_0.npy
    train_data_neg = np.load(dataset_path_neg, allow_pickle=True)
    train_data_neg = list(train_data_neg)
    val_data_neg = np.load(dataset_path_neg, allow_pickle=True)
    val_data_neg = list(val_data_neg)
    
    # for all neg data and neg val data
    # print('neg data is being loaded...')
    # train_data_neg=[]
    # dataset_path_neg_root = '/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data_neg/'
    # file_names = os.listdir(dataset_path_neg_root)
    # neg_data_list = [
    #     dataset_path_neg_root + file_name for file_name in file_names if (file_name.endswith('.npy') and not file_name.endswith('all.npy'))]
    # for item in tqdm.tqdm(neg_data_list):
    #     train_data_neg+=np.load(item, allow_pickle=True).tolist()
    # print('neg val data is being loaded...')
    # val_data_neg=train_data_neg
    
    
    # val_data_neg = list(np.load( '/root/autodl-tmp/new_nde/test_neg_dataset.npy', allow_pickle=True))                
    

    """
    dataset_path_neg1 = '/root/autodl-tmp/data/distill_dataset/AE_mlp_FP_2.npy'
    train_data_neg1 = np.load(dataset_path_neg1, allow_pickle=True)
    train_data_neg1 = list(train_data_neg1)
    train_data_neg += train_data_neg1
    """
    print("train_dataset_neg shape:", len(train_data_neg))
    train_dataset_neg = create_neg_dataset(train_data_neg)
    val_dataset_neg = create_neg_dataset(val_data_neg)
    print(len(train_dataset_neg))

    # Build DataLoader
    # too big batch size will cause loader failing to load data, set to 512
    train_pos_loader = DataLoader(train_dataset_pos,batch_size=512,shuffle=True,drop_last = True)
    train_neg_loader = DataLoader(train_dataset_neg,batch_size=512,shuffle=True,drop_last = True)
    
    val_pos_loader = DataLoader(val_dataset_pos, batch_size=512, shuffle=True)
    val_neg_loader = DataLoader(val_dataset_neg, batch_size=512, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Build Trainer
    trainer = Trainer(args,train_pos_loader,train_neg_loader,val_pos_loader,val_neg_loader)
    # Train & Validate
    old_data_idx = 0
    val_acc_best=0
    best_model_index=0
    for epoch in tqdm.tqdm(range(args.start_epoch,args.epochs)):        
        print('Train Epoch: {}'.format(epoch))
        train_neg_loader = DataLoader(train_dataset_neg,batch_size=512,shuffle=True,drop_last = True)
        trainer.train_neg_loader = train_neg_loader
        train_scores, train_acc = trainer.train(epoch)
        val_scores, val_acc = trainer.validate(epoch)
        if val_acc>val_acc_best:
            val_acc_best = val_acc
            best_model_index = epoch
        trainer.save(epoch, args.output_model_prefix)
    trainer.save_best()
    writer.close()

def test(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    """
    dataset_path_pos = args.dataset_dir + 'test_dataset_pos_new.npy'
    test_data_pos = np.load(dataset_path_pos,allow_pickle=True)
    test_data_pos = list(test_data_pos)
    
    dataset_path_neg_1 = args.dataset_dir + 'test_dataset_neg_new.npy'
    test_data_neg_1 = np.load(dataset_path_neg_1,allow_pickle=True)
    test_data_neg_1 = list(test_data_neg_1)
    
    dataset_path_neg_2 = args.dataset_dir + 'test_dataset_neg_new_2.npy'
    test_data_neg_2 = np.load(dataset_path_neg_2,allow_pickle=True)
    test_data_neg_2 = list(test_data_neg_2)
    
    test_data_neg = test_data_neg_1 + test_data_neg_2
    
    test_dataset = create_dataset_new(test_data_pos, test_data_neg,'val')
    """
    #dataset_path_pos = '/root/autodl-tmp/data/dataset/train_dataset_pos_narrow.npy'
    #dataset_path_pos = '/root/autodl-tmp/data/distill_dataset/train_dataset_pos_narrow_2.npy' # old
    dataset_path_pos = '/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data_pos/transitions_all.npy'
    print('loading pos data...')
    train_data_pos = np.load(dataset_path_pos, allow_pickle=True)
    print('pos data loaded')
    train_data_pos = list(train_data_pos)
    
    #dataset_path_neg = '/root/autodl-tmp/data/distill_dataset/AE_mlp_FP_1.npy'
    #dataset_path_neg = '/root/autodl-tmp/data/distill_dataset/AE_mlp_FP_1.npy'
    #dataset_path_neg = '/root/autodl-tmp/data/dataset/train_dataset_neg_1_all.npy' #old
    
    # for simple neg data test
    dataset_path_neg = '/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data_neg/transitions_0.npy'
    train_data_neg=np.load(dataset_path_neg, allow_pickle=True)
    train_data_neg = list(train_data_neg)

    # for all neg data

    # print('loading neg data...')
    # # train_data_neg = np.load(dataset_path_neg, allow_pickle=True)
    # train_data_neg=[]
    # dataset_path_neg_root = '/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data_neg/'
    # file_names = os.listdir(dataset_path_neg_root)
    # neg_data_list = [
    #     dataset_path_neg_root + file_name for file_name in file_names if (file_name.endswith('.npy') and not file_name.endswith('all.npy'))]
    # for item in tqdm.tqdm(neg_data_list):
    #     train_data_neg+=np.load(item, allow_pickle=True).tolist()
    # 
    # print('neg data loaded')
    
    
    print('pos data shape:', len(train_data_pos), 'neg data shape:', len(train_data_neg))

    print('test_dataset is being created...')
    test_dataset = create_dataset_new(train_data_pos, train_data_neg,0,'val')

    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    model = Reward_Model()
    print('loading model...')
    model.load_state_dict(torch.load('/home/teamcommon/pjy/Bipedal_walker/criticality/stage1/storage/rw_149.pt'))
    model.eval().to(device)
    #model = torch.load('new_log/reward_model/mlp.ep92')
    #model2 = torch.load('new_log/reward_model/mlp.ep132')
    losses, accs, precisions, recalls, TPRs, FPRs = 0, 0, 0, 0, 0, 0
    cls_losses, sc_losses = 0, 0
    n_batches, n_samples = len(test_loader),len(test_loader.dataset)
    #self.train_neg_iter = iter(self.train_neg_loader)
    #model2.eval().to(device)
    
    pos_scores = []
    neg_scores = []
    TP = 0
    FP = 0
    FP_samples = []
    TP_samples = []
    g1 = []
    g2 = []
    g3 = []
    g4 = []
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # test data
            inputs, labels = data[0], data[1]
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels.reshape(-1,2)
            # labels_cls = torch.tensor(labels[:,0],dtype=torch.int64)
            # 修改后（消除警告，安全复制）（豆包）
            labels_cls = labels[:,0].clone().detach().to(dtype=torch.int64)
            labels_dist = labels[:,1]
            bs = inputs.shape[0]

            outputs = model(inputs)
            #outputs2 = model2(inputs)

            rewards = outputs["rewards"]
            #rewards2 = outputs2["rewards"]
            for k in range(bs):
                cur_input = inputs[k].squeeze(0).clone().detach().cpu().numpy()
                reward = rewards[k,-1].item()
                #reward2 = rewards2[k,-1].item()
                label = labels_cls[k].item()
                
                reward_threshold = 2.5

                """
                if reward > reward_threshold:
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
                
            
                
                if label == 1:
                    pos_scores.append(reward)
                else:
                    neg_scores.append(reward)
            
                
    print(TP,FP)
    """
    print(min(g1),max(g1))
    print(min(g2),max(g2))
    print(min(g3),max(g3))
    print(min(g4),max(g4))
    """
    #np.save(f'/root/autodl-tmp/data/distill_dataset/all_rm_FP_1.npy', FP_samples, allow_pickle=True)
    #np.save(f'/root/autodl-tmp/data/distill_dataset/rm_TP_2.npy', TP_samples, allow_pickle=True)

    return pos_scores, neg_scores

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
    plt.scatter(range(len(loss_neg)), 
                loss_neg,  
                alpha=0.7, 
                marker=markers[1], 
                c=colors[1], 
                label=labels[1])
    
    #plt.subplot(121)
    plt.scatter([k * (len(loss_neg)/len(loss_pos)) for k in range(len(loss_pos))], 
                loss_pos,  
                alpha=0.7, 
                marker=markers[0], 
                c=colors[0], 
                label=labels[0])
    
    plt.title('Scores')
    plt.ylabel('Scores'); plt.xlabel('Index')
    plt.savefig('/home/teamcommon/pjy/Bipedal_walker/criticality/stage1/storage/reward_model_test.png')

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

# run model to get FT samples
def run_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Reward_Model()
    # load the last mmodel
    print('loading model...')
    model.load_state_dict(torch.load('/home/teamcommon/pjy/Bipedal_walker/criticality/stage1/storage/rw_149.pt'))
    model.eval().to(device)

    print('neg data is being loaded...')
    data_neg=[]
    dataset_path_neg_root = '/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data_neg_new/'
    file_names = os.listdir(dataset_path_neg_root)
    neg_data_list = [
        dataset_path_neg_root + file_name for file_name in file_names if (file_name.endswith('.npy') and not file_name.endswith('all.npy'))]
    for item in tqdm.tqdm(neg_data_list):
        data_neg+=np.load(item, allow_pickle=True).tolist()
    
    dataset_neg = create_neg_dataset(data_neg)

    # 2048似乎是一个上限，也不完全，取决于其它正在运行的进程
    test_neg_loader = DataLoader(dataset_neg, batch_size=1024, shuffle=True,drop_last = True)

    reward_threshold=-0.111
    
    FP_list=[]

    for i,data in tqdm.tqdm(enumerate(test_neg_loader)):
        # neg data
        input, label = data[0],data[1]
        inputs=input
        labels=label
        # for debug
        # print(label1[:,0].sum(), label2[:,0].sum())
        # for debug
        # print(f'training, inputs shape is {inputs.shape}')
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels.reshape(-1,2)
        labels_cls = torch.tensor(labels[:,0],dtype=torch.int64).clone().detach()
        labels_dist = labels[:,1]
        bs = inputs.shape[0]
        outputs = model(inputs)
        rewards = outputs["rewards"]
        # for debug
        # print(label1)
        # print(label2)
        # print(labels_cls)
        max_reward =rewards[:, -1].max()
        if max_reward<20:
            max_reward =20
        min_reward =rewards[:, -1].min()
        if min_reward>-20:
            min_reward =-20
        mid_reward = (max_reward + min_reward) / 2
        med_reward =median(rewards[:,-1])
        reward_threshold_new = mid_reward + reward_threshold * (max_reward - min_reward) * 0.5
        # print(f'Reward threshold new is {reward_threshold_new}')
        for k in range(bs):
            reward = rewards[k,-1].item()
            #reward2 = rewards2[k,-1].item()
            label = labels_cls[k].item()
            if reward > reward_threshold_new:
                # FP
                FP_list.append(data_neg[i*512+k])
            else:
                # TN
                pass
    
    save_path='../data/processed_by_stage1/FP_samples.npy'
    np.save(save_path, FP_list, allow_pickle=True)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input parameters
    # 2048
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--output_model_prefix', default='mlp')

    # Train parameters
    parser.add_argument('--start_epoch', default=101, type=int, help='the number of epochs')
    parser.add_argument('--epochs', default=150, type=int, help='the number of epochs')
    # 1e-4
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--final_ratio',default=0,type=int,help='ratio of negative / positive')
    parser.add_argument('--start_ratio',default=0,type=int, help='ratio of negative / positive')
    parser.add_argument('--warm_up_epochs',default=20,type=int)
    # 0.546
    parser.add_argument('--threshold', default=0.5)

    # Model parameters
    parser.add_argument('--input_dim', default=25, type=int, help='the number of classes')
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
    parser.add_argument('--dataset_dir',default='/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new') # 这个好像没有用
    parser.add_argument('--log_dir', default='/home/teamcommon/pjy/Bipedal_walker/criticality/log/log')
    parser.add_argument('--is_resume',default=0)
    parser.add_argument('--is_train',default=0)
 
    # parser.add_argument('--model_path',default='new_log/model_new/mlp.ep99')
    # parser.add_argument('--best_model_path',default='new_log/model/mlp.ep680')
    # parser.add_argument('--first_model_path',default='new_log/model_new/mlp.ep546')
    parser.add_argument('--model_path',default='/home/teamcommon/pjy/Bipedal_walker/criticality/stage1/storage/rw_149.pt')
    parser.add_argument('--best_model_path',default='/home/teamcommon/pjy/Bipedal_walker/criticality/log/model/mlp.ep95')
    parser.add_argument('--first_model_path',default='/home/teamcommon/pjy/Bipedal_walker/criticality/log/model/mlp.ep95')
    parser.add_argument('--two_model',default=0)

    args = parser.parse_args()
    
    main_new(args)
    run_model()

    # print('is_train:{}', args.is_train)
    # if args.is_train:
    #     main_new(args)
    # else:
    #     pos, neg = test_new(args)
        

