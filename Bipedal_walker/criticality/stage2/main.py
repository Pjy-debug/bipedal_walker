import sys
import os
# 获取当前文件的绝对路径
current_path = os.path.abspath(__file__)
# 向上一级目录
parent_dir = os.path.dirname(os.path.dirname(current_path))
# 将criticality目录添加到Python的模块搜索路径
sys.path.append(parent_dir)

# 现在可以导入criticality_目录下的函数了
from data_.data_utils import create_dataset,create_dataset_new,create_dataset_new_2, create_pos_dataset, create_neg_dataset

import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from trainer import Trainer
from test import test

import tqdm


def main(args):
    print(args)
    writer = SummaryWriter(log_dir=args.log_dir)
    train_info = []
    val_info = []
    test_info = []

    # build dataset
    old_ratio = args.start_ratio + (args.final_ratio - args.start_ratio)*args.start_epoch/args.warm_up_epochs
    old_ratio = int(old_ratio)
    print(f'ratio is {old_ratio},loading data...')
    
    dataset_path_pos = args.dataset_dir + 'train_dataset_pos.npy'
    train_data_pos = np.load(dataset_path_pos, allow_pickle=True)
    train_data_pos = list(train_data_pos)
    
    #dataset_path_neg = '/root/brx/tta_new/tta/criticality/new_log/result/train_dataset_neg_6.npy'
    #dataset_path_neg = args.dataset_dir + f'train_dataset_neg_{9}.npy' 
    dataset_path_neg = '/root/autodl-tmp/data/dataset/distilled_train_dataset_neg_199.npy'
    train_data_neg = list(np.load(dataset_path_neg, allow_pickle=True))
    print(len(train_data_neg))
   
    train_dataset = create_dataset_new(train_data_pos, train_data_neg, 0,'train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset = create_dataset(args.dataset_dir + 'val_dataset_pos.npy',args.dataset_dir + 'val_dataset_neg.npy', old_ratio,'val')
    # build dataset
    test_dataset = create_dataset(args.dataset_dir + 'test_dataset_pos.npy',args.dataset_dir + 'test_dataset_neg.npy', 0, 'test')

    print(f'Successfully build train dataset! Length of train dataset is {len(train_dataset)}')
    print(f'Successfully build val dataset! Length of val dataset is {len(val_dataset)}')
    print(f'Successfully build test dataset! Length of test dataset is {len(test_dataset)}')

    # Build DataLoader
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Build Trainer
    trainer = Trainer(args,train_loader,val_loader)
    best_acc = 0
    # Train & Validate
    old_data_idx = 0
    for epoch in range(args.start_epoch,args.epochs):
        if epoch < args.warm_up_epochs:
            ratio = args.start_ratio + (args.final_ratio - args.start_ratio)*epoch/args.warm_up_epochs
            ratio = int(ratio)
        else:
            ratio = args.final_ratio
        if ratio != old_ratio:
            # build dataset
            print('building dataset')
            train_dataset = create_dataset(args.dataset_dir + 'train_dataset_pos.npy', args.dataset_dir + 'train_dataset_neg.npy', ratio)
            val_dataset = create_dataset(args.dataset_dir + 'test_dataset_pos.npy', args.dataset_dir + 'test_dataset_neg.npy', ratio)
            # Build DataLoader
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
            trainer.train_loader = train_loader
            trainer.test_loader = val_loader
        
        old_ratio = ratio
        train_loss, train_acc, train_precision, train_recall, train_TPR, train_FPR = trainer.train(epoch)
        val_loss, val_acc, val_precision, val_recall, val_TPR, val_FPR = trainer.validate(epoch)
        trainer.save(epoch, args.output_model_prefix)
        if val_acc > best_acc:
            trainer.save_best()
            best_acc = val_acc
        
        args.model_path= 'new_log/model/mlp.ep%d'% epoch
        test_loss, test_acc, test_precision, test_recall, test_TPR, test_FPR = test(args,test_loader)
        test_info.append({'train_loss':test_loss,
                          'train_acc':test_acc,
                          'train_precision':test_precision,
                          'train_recall':test_recall,
                          'train_TPR':test_TPR,
                          'train_FPR':test_FPR,
                          'epoch':epoch})
        
        
        train_info.append({'train_loss':train_loss,
                          'train_acc':train_acc,
                          'train_precision':train_precision,
                          'train_recall':train_recall,
                          'train_TPR':train_TPR,
                          'train_FPR':train_FPR,
                          'epoch':epoch})
        val_info.append({'train_loss':val_loss,
                          'train_acc':val_acc,
                          'train_precision':val_precision,
                          'train_recall':val_recall,
                          'train_TPR':val_TPR,
                          'train_FPR':val_FPR,
                          'epoch':epoch})

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', val_acc, epoch)
        writer.add_scalar('Precision/train', train_precision, epoch)
        writer.add_scalar('Precision/test', val_precision, epoch)
        writer.add_scalar('Recall/train', train_recall, epoch)
        writer.add_scalar('Recall/test', val_recall, epoch)
        writer.add_scalar('ROC/train', train_TPR, train_FPR)
        writer.add_scalar('ROC/test', val_TPR, val_FPR)

    writer.close()
    np.save(f'/root/brx/tta_new/tta/criticality/new_log/result/train_info_{epoch}.npy', train_info, allow_pickle=True)
    np.save(f'/root/brx/tta_new/tta/criticality/new_log/result/val_info_{epoch}.npy', val_info, allow_pickle=True)
    np.save(f'/root/brx/tta_new/tta/criticality/new_log/result/test_info_{epoch}.npy', test_info, allow_pickle=True)

def data_load_and_divide_712():
    dataset_path_pos = '/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data_pos_new/transitions_all.npy'
    print('pos data is being loaded...')
    data_pos = np.load(dataset_path_pos, allow_pickle=True)

    # for all neg data and neg val data
    dataset_path_neg='/home/teamcommon/pjy/Bipedal_walker/criticality/data/processed_by_stage1/FP_samples.npy'
    print('neg data is being loaded...')
    data_neg=np.load(dataset_path_neg, allow_pickle=True)

    # for debug
    print(f'len of data pos is {len(data_pos)}, len of data neg is {len(data_neg)}')
     
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

    # build dataset
    old_ratio = args.start_ratio + (args.final_ratio - args.start_ratio)*args.start_epoch/args.warm_up_epochs
    old_ratio = int(old_ratio)
    print(f'ratio is {old_ratio},loading data...')
    
    # 注意：model的input_dim是107
    [train_data_pos, val_data_pos, test_data_pos, train_data_neg, val_data_neg, test_data_neg]=data_load_and_divide_712()
   
    train_data_neg_len=len(train_data_neg)
    train_data_neg_len_1=round(train_data_neg_len//3)
    train_data_neg_len_2=train_data_neg_len_1*2
    # 对于trainer，需要三个dataset：train_pos_dataset，train_neg_dataset1，train_neg_dataset2，大小[1:1:2]
    train_pos_dataset = create_pos_dataset(train_data_pos)
    train_neg_dataset1 = create_neg_dataset(train_data_neg[0:train_data_neg_len_1])
    train_neg_dataset2 = create_neg_dataset(train_data_neg[train_data_neg_len_1:train_data_neg_len_1+train_data_neg_len_2])
    
    val_dataset = create_dataset_new(val_data_pos, val_data_neg, 0,'val')
    # build dataset
    test_dataset = create_dataset_new(test_data_pos, test_data_neg, 0, 'test')

    print(f'Successfully build train datasets!')
    print(f'Successfully build val datasets!')
    print(f'Successfully build test datasets!')

    # Build DataLoader
    train_pos_loader = DataLoader(train_pos_dataset, batch_size=args.batch_size, shuffle=True)
    train_neg_loader1 = DataLoader(train_neg_dataset1, batch_size=args.batch_size, shuffle=True)
    train_neg_loader2 = DataLoader(train_neg_dataset2, batch_size=args.batch_size*2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Build Trainer
    trainer = Trainer(args,train_loader,val_loader)
    best_acc = 0
    # Train & Validate
    old_data_idx = 0
    for epoch in range(args.start_epoch,args.epochs):
        if epoch < args.warm_up_epochs:
            ratio = args.start_ratio + (args.final_ratio - args.start_ratio)*epoch/args.warm_up_epochs
            ratio = int(ratio)
        else:
            ratio = args.final_ratio
        if ratio != old_ratio:
            # build dataset
            print('building dataset')
            train_dataset = create_dataset(args.dataset_dir + 'train_dataset_pos.npy', args.dataset_dir + 'train_dataset_neg.npy', ratio)
            val_dataset = create_dataset(args.dataset_dir + 'test_dataset_pos.npy', args.dataset_dir + 'test_dataset_neg.npy', ratio)
            # Build DataLoader
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
            trainer.train_loader = train_loader
            trainer.test_loader = val_loader
        
        old_ratio = ratio
        train_loss, train_acc, train_precision, train_recall, train_TPR, train_FPR = trainer.train(epoch)
        val_loss, val_acc, val_precision, val_recall, val_TPR, val_FPR = trainer.validate(epoch)
        trainer.save(epoch, args.output_model_prefix)
        if val_acc > best_acc:
            trainer.save_best()
            best_acc = val_acc
        
        args.model_path= 'new_log/model/mlp.ep%d'% epoch
        test_loss, test_acc, test_precision, test_recall, test_TPR, test_FPR = test(args,test_loader)
        test_info.append({'train_loss':test_loss,
                          'train_acc':test_acc,
                          'train_precision':test_precision,
                          'train_recall':test_recall,
                          'train_TPR':test_TPR,
                          'train_FPR':test_FPR,
                          'epoch':epoch})
        
        
        train_info.append({'train_loss':train_loss,
                          'train_acc':train_acc,
                          'train_precision':train_precision,
                          'train_recall':train_recall,
                          'train_TPR':train_TPR,
                          'train_FPR':train_FPR,
                          'epoch':epoch})
        val_info.append({'train_loss':val_loss,
                          'train_acc':val_acc,
                          'train_precision':val_precision,
                          'train_recall':val_recall,
                          'train_TPR':val_TPR,
                          'train_FPR':val_FPR,
                          'epoch':epoch})

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', val_acc, epoch)
        writer.add_scalar('Precision/train', train_precision, epoch)
        writer.add_scalar('Precision/test', val_precision, epoch)
        writer.add_scalar('Recall/train', train_recall, epoch)
        writer.add_scalar('Recall/test', val_recall, epoch)
        writer.add_scalar('ROC/train', train_TPR, train_FPR)
        writer.add_scalar('ROC/test', val_TPR, val_FPR)

    writer.close()
    np.save(f'/root/brx/tta_new/tta/criticality/new_log/result/train_info_{epoch}.npy', train_info, allow_pickle=True)
    np.save(f'/root/brx/tta_new/tta/criticality/new_log/result/val_info_{epoch}.npy', val_info, allow_pickle=True)
    np.save(f'/root/brx/tta_new/tta/criticality/new_log/result/test_info_{epoch}.npy', test_info, allow_pickle=True)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input parameters
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--output_model_prefix', default='mlp')

    # Train parameters
    parser.add_argument('--start_epoch', default=640, type=int, help='the number of epochs')
    parser.add_argument('--epochs', default=660, type=int, help='the number of epochs')
    # 1e-4
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--final_ratio',default=0,type=int,help='ratio of negative / positive')
    parser.add_argument('--start_ratio',default=0,type=int, help='ratio of negative / positive')
    parser.add_argument('--warm_up_epochs',default=20,type=int)
    parser.add_argument('--threshold', default=0.6)

    # Model parameters
    # 105/512/256/1024/6/8/0.1/1024/2/11
    # 105/256/256/1024/3/8/0.5/1024/2/11
    parser.add_argument('--input_dim', default=105, type=int, help='the number of classes')
    parser.add_argument('--embed_dim', default=512, type=int, help='the number of expected features in the mlp')
    parser.add_argument('--embed_dim_1', default=256, type=int, help='the number of expected features in the mlp')
    parser.add_argument('--embed_dim_2', default=1024, type=int, help='the number of expected features in the mlp')
    parser.add_argument('--n_layers', default=6, type=int, help='the number of heads in the multi-head attention network')
    parser.add_argument('--n_heads', default=8, type=int, help='the number of multi-head attention heads')
    parser.add_argument('--dropout', default=0.5, type=float, help='the residual dropout value')
    parser.add_argument('--ffn_dim', default=1024, type=int, help='the dimension of the feedforward network')
    parser.add_argument('--num_classes', default=2, type=int, help='the number of classes')
    parser.add_argument('--max_seq_len', default=11, type=int, help='the number of expected features in the mlp')

    # path
    # 111 133 / 158
    # 31 / 52 50 49 / 71 78 79
    # 8/19/23/32/49/
    # 8.1: 447/443/439
    parser.add_argument('--dataset_dir',default='/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data_pos_new/')
    parser.add_argument('--log_dir', default='new_log/log')
    parser.add_argument('--is_resume',default=0)
    parser.add_argument('--model_path',default='new_log/model/mlp.ep634')
    parser.add_argument('--best_model_path',default='new_log/model/mlp.ep634')

    args = parser.parse_args()

    main(args)