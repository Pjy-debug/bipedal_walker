import sys
import os
# 获取当前文件的绝对路径
current_path = os.path.abspath(__file__)
# 向上一级目录
parent_dir = os.path.dirname(os.path.dirname(current_path))
# 将criticality目录添加到Python的模块搜索路径
sys.path.append(parent_dir)

# 现在可以导入criticality_目录下的函数了
from data_.data_utils import create_dataset,create_dataset_new,create_pos_dataset,create_neg_dataset

import numpy as np
import csv
import matplotlib.pyplot as plt
from test import test

import argparse
from torch.utils.data import DataLoader



def picture(x,y,label,save_path):
    plt.plot(x,y,label=label)
    #plt.plot([0,1],[0,0.7],linestyle='dashed')
    plt.title(label)
    plt.xlabel('epoch')
    plt.ylabel(label)
    plt.legend(loc=1)
    #plt.xlim((0,1))
    #plt.ylim((0,1))
    plt.show()
    plt.savefig(save_path)

def draw_roc(x,y, save_path='new_log/result/cur_roc_789.png'):
    plt.plot(x,y)
    plt.plot([0,1],[0,1],linestyle='dashed')
    plt.title('ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    #plt.legend(loc=1)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.show()
    plt.savefig(save_path)

def draw_pr(x,y, save_path='new_log/result/cur_pr_789.png'):
    y_new = []

    y_new=y.copy()
    # # 不知道这里在干什么
    # for k in range(len(y)):
    #     item = y[k]
    #     if 8 < k < len(y)-1:
    #         y_new.append(item + 0.1)
    #     else:
    #         y_new.append(item)
    plt.plot(x,y_new)
    plt.plot([0,1],[1,0],linestyle='dashed')
    plt.title('Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.legend(loc=1)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.show()
    plt.savefig(save_path)

def roc(args):
    # 'tf-logs/result/train_loss.png'
    print('loading data...')
    test_dataset = create_dataset(args.dataset_dir + 'test_dataset_pos.npy',args.dataset_dir + 'test_dataset_neg.npy', 0, 'test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    print('successfully load data!')
    roc_x = []
    roc_y = []
    pr_x = []
    pr_y = []
    thresholds = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
    #thresholds = [0.37,0.371,0.372,0.373,0.374,0.375,0.376,0.377,0.378,0.379,0.38]
    for threshold in thresholds:
        print(threshold)
        args.threshold = threshold
        
        """
        for x in thresholds:
            test_loss, test_acc, test_precision, test_recall, test_TPR, test_FPR = test(args,test_loader,x)
            print(x,test_acc,test_precision,test_recall,test_TPR,test_FPR)
        
        """
        test_loss, test_acc, test_precision, test_recall, test_TPR, test_FPR = test(args,test_loader)
        print(test_acc,test_precision,test_recall,test_TPR,test_FPR)
        
        
        roc_x.append(test_FPR)
        roc_y.append(test_TPR)
        pr_x.append(test_recall)
        pr_y.append(test_precision)
        
    test_info = {'roc_x':roc_x,'roc_y':roc_y,'pr_x':pr_x,'pr_y':pr_y,}
    np.save(f'/root/brx/tta_new/tta/criticality/new_log/result/cur_roc_pr_789.npy', test_info, allow_pickle=True)
    return roc_x,roc_y,pr_x,pr_y
    

    
def read_data(data_paths,save_path):
    # 'tf-logs/result/train_loss.png'
    
    x = []
    loss = []
    acc = []
    precision = []
    recall = []
    FPR = []
    TPR = []
    roc = []
    pr = []
    for data_path in data_paths:
        print(data_path)
        train_info = np.load(data_path, allow_pickle=True)
        #print(len(train_info))
        for item in train_info:
            x.append(item['epoch'])
            loss.append(item['train_loss'])
            precision.append(item['train_precision'])
            recall.append(item['train_recall'])
            acc.append(item['train_acc'])
            FPR.append(item['train_FPR'])
            TPR.append(item['train_TPR'])
            print(item['epoch'],item['train_loss'],item['train_precision'],item['train_acc'],item['train_recall'])
    
    #picture(range(len(precision)),precision,'precision',save_path)
    picture(range(len(recall)),recall,'recall',save_path)
    #picture(range(len(acc)),acc,'accuracy',save_path)
    #picture(range(len(loss)),loss,'loss',save_path)
    
"""
data_path = [
             'new_log/result/test_info_449.npy',]

data_path = ['new_log/result/train_info_139.npy',
             'new_log/result/train_info_239.npy','new_log/result/train_info_259.npy',
             'new_log/result/train_info_299.npy','new_log/result/train_info_319.npy',
             'new_log/result/train_info_339.npy','new_log/result/train_info_359.npy']

"""
data_path = ['new_log/result/val_info_139.npy',
             'new_log/result/val_info_239.npy','new_log/result/val_info_259.npy',
             'new_log/result/val_info_299.npy','new_log/result/val_info_319.npy',
             'new_log/result/val_info_339.npy','new_log/result/val_info_359.npy']

            

#save_path = 'new_log/result/val_recall.png'
#read_data(data_path,save_path)


parser = argparse.ArgumentParser()

# Input parameters
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--output_model_prefix', default='mlp')

# Train parameters
parser.add_argument('--start_epoch', default=340, type=int, help='the number of epochs')
parser.add_argument('--epochs', default=360, type=int, help='the number of epochs')
# 1e-4
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--final_ratio',default=0,type=int,help='ratio of negative / positive')
parser.add_argument('--start_ratio',default=0,type=int, help='ratio of negative / positive')
parser.add_argument('--warm_up_epochs',default=20,type=int)
parser.add_argument('--threshold', default=0.5)

# Model parameters
parser.add_argument('--input_dim', default=105, type=int, help='the number of classes')
parser.add_argument('--embed_dim', default=512, type=int, help='the number of expected features in the mlp')
parser.add_argument('--embed_dim_1', default=256, type=int, help='the number of expected features in the mlp')
parser.add_argument('--embed_dim_2', default=1024, type=int, help='the number of expected features in the mlp')
parser.add_argument('--n_layers', default=6, type=int, help='the number of heads in the multi-head attention network')
parser.add_argument('--n_heads', default=8, type=int, help='the number of multi-head attention heads')
parser.add_argument('--dropout', default=0.1, type=float, help='the residual dropout value')
parser.add_argument('--ffn_dim', default=1024, type=int, help='the dimension of the feedforward network')
parser.add_argument('--num_classes', default=2, type=int, help='the number of classes')
parser.add_argument('--max_seq_len', default=11, type=int, help='the number of expected features in the mlp')

# path
# 111 133 / 158
# 31 / 52 50 49 / 71 78 79
# 8/19/23/32/49/
# 108/114/142/147/151/167/171/189/224/230/233/242/246/259/278/292/305/338
parser.add_argument('--dataset_dir',default='/root/autodl-tmp/data/dataset/')
parser.add_argument('--log_dir', default='new_log/log')
parser.add_argument('--is_resume',default=1)
# 'new_log/model/mlp.ep142'
parser.add_argument('--model_path',default='new_log/model_new/mlp.ep789')
parser.add_argument('--first_model_path',default='new_log/model_new/mlp.ep546')
parser.add_argument('--two_model',default=1)

args = parser.parse_args()

#roc_x,roc_y,pr_x,pr_y=roc(args)
#draw_roc(roc_x,roc_y)

train_info = np.load('/root/brx/tta_new/tta/criticality/new_log/result/cur_roc_pr_789.npy', allow_pickle=True)
train_info = dict(train_info.tolist())
pr_x = train_info['pr_x']
pr_y = train_info['pr_y']
draw_pr(pr_x,pr_y)
    
  
        
        