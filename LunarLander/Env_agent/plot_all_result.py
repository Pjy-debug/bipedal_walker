import numpy as np
import csv
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from torch.utils.data import DataLoader

from utils.data_utils import create_dataset_new, create_episode_dataset
from test_all_stages import test


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

def draw_roc(x,y):
    plt.plot(x,y,label='roc')
    plt.plot([0,1],[0,1],linestyle='dashed')
    plt.title('roc')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc=1)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.show()
    plt.savefig('new_log/result/cur_roc.png')

def draw_pr(x,y):
    plt.plot(x,y)
    plt.plot([0,1],[1,0],linestyle='dashed')
    plt.title('precision-recall')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend(loc=1)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.show()
    plt.savefig('new_log/result/cur_pr.png')
    
def draw_roc_new(test_info):
    for item in test_info:
        x = item['roc_x']
        y = item['roc_y']
        label = item['test_ratio']
        plt.plot(x,y,label=f'ratio of test {label}')
    
    plt.plot([0,1],[0,1],linestyle='dashed') 
    plt.title('roc')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc=1)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.show()
    plt.savefig('new_log/result/cur_roc_450.png')

def draw_pr_new(test_info):
    for item in test_info:
        x = item['pr_x']
        #print(x)
        y = item['pr_y']
        for k in range(len(x)):
            if not x[k]:
                y[k] = 1.0
        label = item['test_ratio']
        plt.plot(x,y,label=f'ratio of test {label}')
    
    plt.plot([0,1],[1,0],linestyle='dashed')
    plt.title('precision-recall')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend(loc=1)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.show()
    plt.savefig('result/cur_pr_450.png')

def roc(args):
    # 'tf-logs/result/train_loss.png'
    #test_ratio = [0,10,50,100,150,200,500,1000,5000,10000]
    """
    dataset_path_pos ='/mnt1/brx/dataset/test_dataset_pos.npy'
    #dataset_path_pos = '/root/autodl-tmp/data/distill_dataset/train_dataset_pos_narrow_2.npy'
    test_data_pos = np.load(dataset_path_pos,allow_pickle=True)
    test_data_pos = list(test_data_pos)
    
    dataset_path_neg_1 = '/mnt1/brx/dataset/test_dataset_neg.npy'
    test_data_neg_1 = np.load(dataset_path_neg_1,allow_pickle=True)
    test_data_neg_1 = list(test_data_neg_1)
    
    test_data_neg = test_data_neg_1 
    
    test_dataset = create_dataset_new(test_data_pos, test_data_neg,'train')
    """
    
    dataset_path_pos ='/mnt1/brx/dataset/test_dataset_pos_episode_450_new.npy'
    test_data_pos = np.load(dataset_path_pos,allow_pickle=True)
    test_data_pos = list(test_data_pos)
    
    dataset_path_neg_1 = '/mnt1/brx/dataset/test_dataset_neg_episode_450_new.npy'
    test_data_neg_1 = np.load(dataset_path_neg_1,allow_pickle=True)
    test_data_neg_1 = list(test_data_neg_1)
    test_data_neg = test_data_neg_1 
    
    test_dataset = create_episode_dataset(test_data_pos, test_data_neg,'total')
    
    # Build DataLoader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    roc_x = []
    roc_y = []
    pr_x = []
    pr_y = []
    
    thresholds = [0.02*k for k in range(0,50)]
    for threshold in thresholds:
        print(threshold)
        args.threshold = threshold
        test_loss, test_acc, test_precision, test_recall, test_TPR, test_FPR = test(args,test_loader)
        print(test_acc,test_precision,test_recall,test_TPR,test_FPR)
        roc_x.append(test_FPR)
        roc_y.append(test_TPR)
        pr_x.append(test_recall)
        pr_y.append(test_precision)

    test_info_ = {'roc_x':roc_x,'roc_y':roc_y,'pr_x':pr_x,'pr_y':pr_y}
    test_info_.append(test_info_)
    
    np.save(f'result/cur_roc_pr_450.npy', test_info_, allow_pickle=True)
    print('successfully save test info!')
    # return roc_x,roc_y,pr_x,pr_y
    

    
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
parser.add_argument('--dataset_dir',default='/mnt1/data/dataset/')
parser.add_argument('--log_dir', default='new_log/log')
parser.add_argument('--is_resume',default=1)
# 'new_log/model/mlp.ep142'
parser.add_argument('--model_path',default='new_log/model/mlp.ep160')
parser.add_argument('--best_model_path',default='new_log/model/mlp.ep142')
parser.add_argument('--two_model',default=0)

args = parser.parse_args()

#roc_x,roc_y,pr_x,pr_y=roc(args)
#draw_roc(roc_x,roc_y)
roc(args)
train_info = np.load('result/cur_roc_pr_450.npy', allow_pickle=True)
#train_info = dict(train_info.tolist())
train_info = train_info.tolist()
pr_x = train_info['pr_x']
pr_y = train_info['pr_y']
draw_pr_new(train_info)
#draw_roc_new(train_info)

  
        
        