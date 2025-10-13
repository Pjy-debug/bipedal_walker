import argparse
from torch.utils.data import DataLoader,Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random

from utils.data_utils import create_dataset_new,create_pos_dataset,create_neg_dataset,create_distill_dataset,create_episode_dataset,create_episode_distill_dataset
from trainer import Trainer


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
    
    dataset_path_pos = '/mnt/mnt1/tyy/data/positive/train_pos.npy'
    #dataset_path_pos = '/mnt/mnt1/tyy/data/positive/pos_450_4_1.npy'
    train_data_pos = np.load(dataset_path_pos, allow_pickle=True)
    train_data_pos = list(train_data_pos)
    # --- 新增的代码: 选取 10% 的正样本数据 ---
    random_seed = 42  
    random.seed(random_seed)
    random.shuffle(train_data_pos)  # 随机打乱正样本数据
    train_data_pos = train_data_pos[:len(train_data_pos) // 10]
    
    train_dataset_pos = create_episode_dataset(train_data_pos, [], 'pos')
    
    print("pos len", len(train_data_pos))
    
    test_dataset_path_pos ='/mnt/mnt1/tyy/data/positive/val_pos.npy'
    #test_dataset_path_pos ='/mnt/mnt1/tyy/data/positive/pos_450_4_2.npy'
    test_data_pos = np.load(test_dataset_path_pos,allow_pickle=True)
    test_data_pos = list(test_data_pos)

    #新增 取10%的pos
    random_seed = 42  
    random.seed(random_seed)
    random.shuffle(test_data_pos)  # 随机打乱正样本数据
    test_data_pos = test_data_pos[:len(test_data_pos) // 10]
    
    #dataset_path_neg1 = '/mnt/mnt1/tyy/data/newnegative/train_neg.npy'
    dataset_path_neg1 = '/home/teamcommon/tyy/MyLander/Env_agent/stage1/40new_train_FP_final5.npy'
    train_data_neg1 = np.load(dataset_path_neg1, allow_pickle=True)
    train_data_neg = list(train_data_neg1)

    train_dataset_neg = create_episode_distill_dataset([],train_data_neg,'neg')
    #train_dataset_neg = create_episode_dataset([],train_data_neg,'neg')
    #train_dataset_neg = create_neg_dataset(train_data_neg)
    print(f'Successfully build neg dataset! Length of neg dataset is {len(train_dataset_neg)}')
    

    #test_dataset_path_neg_1 = '/mnt/mnt1/tyy/data/newnegative/val_neg.npy'
    #test_dataset_path_neg_1 = '/mnt/mnt1/tyy/data/newnegative/neg_450_4_1.npy'
    test_dataset_path_neg_1 = '/home/teamcommon/tyy/MyLander/Env_agent/stage1/40new_val_FP_final5.npy'
    test_data_neg_1 = np.load(test_dataset_path_neg_1,allow_pickle=True)
    test_data_neg_1 = list(test_data_neg_1)
    
    test_data_neg = test_data_neg_1 
    
    train_dataset2 = create_episode_distill_dataset(train_data_pos,train_data_neg,'total')
    #train_dataset2 = create_episode_distill_dataset(train_dataset_pos_subset,train_data_neg,'total')
    #val_dataset = create_episode_dataset(test_data_pos,test_data_neg,'total')
    val_dataset = create_episode_distill_dataset(test_data_pos,test_data_neg,'total')
    
    """
    val_dataset = create_dataset(args.dataset_dir + 'val_dataset_pos.npy',args.dataset_dir + 'val_dataset_neg.npy', 0,'val',5000)
    """

    print(f'Successfully build val dataset! Length of val dataset is {len(val_dataset)}')
    #print(f'Successfully build test dataset! Length of test dataset is {len(test_dataset)}')

    # Build DataLoader
    train_pos_loader = DataLoader(train_dataset_pos,batch_size=args.batch_size * 100 ,shuffle=True,drop_last = True)
    #train_pos_loader = DataLoader(train_dataset_pos_subset,batch_size=args.batch_size * 100 ,shuffle=True,drop_last = True)
    train_neg_loader = DataLoader(train_dataset_neg,batch_size=args.batch_size* 100 ,shuffle=True,drop_last = True)
    train_neg_loader2 = DataLoader(train_dataset2,batch_size=args.batch_size * 200 ,shuffle=True,drop_last = True)
    val_loader = DataLoader(val_dataset, batch_size= 1024 , shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=args.batch_size , shuffle=True)

    # Build Trainer
    trainer = Trainer(args,train_pos_loader,train_neg_loader,train_neg_loader2, val_loader)
    best_acc = 0
    # Train & Validate
    old_data_idx = 0
    for epoch in range(args.start_epoch,args.epochs):
        print(epoch)
        #用的是FP的neg
        train_neg_loader = DataLoader(train_dataset_neg,batch_size=args.batch_size * 100,shuffle=True,drop_last = True)
        trainer.train_neg_loader1 = train_neg_loader
        #又有neg又有pos
        train_neg_loader2 = DataLoader(train_dataset2,batch_size=args.batch_size * 200,shuffle=True,drop_last = True)
        trainer.train_neg_loader2 = train_neg_loader2

        train_loss, train_acc, train_precision, train_recall = trainer.train(epoch)
        val_loss, val_acc, val_precision, val_recall, val_TPR, val_FPR = trainer.validate(epoch)
        trainer.save(epoch, args.output_model_prefix)
        if val_acc > best_acc:
            trainer.save_best()
            best_acc = val_acc
            
    writer.close()
    #np.save(f'/root/brx/tta_new/tta/criticality/new_log/result_new/train_info_{epoch}.npy', train_info, allow_pickle=True)
    #np.save(f'/root/brx/tta_new/tta/criticality/new_log/result_new/val_info_{epoch}.npy', val_info, allow_pickle=True)
    #np.save(f'/root/brx/tta_new/tta/criticality/new_log/result_new/test_info_{epoch}.npy', test_info, allow_pickle=True)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input parameters
    # 2048
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--output_model_prefix', default='mlp')

    # Train parameters
    parser.add_argument('--start_epoch', default=0, type=int, help='the number of epochs')
    parser.add_argument('--epochs', default=50, type=int, help='the number of epochs')
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
    # 111 133 / 158
    # 31 / 52 50 49 / 71 78 79
    # 8/19/23/32/49/
    # 108/114/142/147/151/167/171/189/224/230/233/242/246/259/278/292/305/338
    # 668
    parser.add_argument('--dataset_dir',default='/mnt1/brx/dataset/')
    parser.add_argument('--log_dir', default='new_log/log')
    #parser.add_argument('--log_dir', default='/home/teamcommon/tyy/MyLander/Env_agent/stage2/picture', help='Directory to save plots')
    #默认is_resume = 0
    parser.add_argument('--is_resume',default=0)
    parser.add_argument('--is_chuan',default=0)
    # 619
    # 865,868,880,887
    # 1087: 提炼过的平衡数据集上训练出来的
    # 113 116 127
    # 负样本特征学习已经到达39
    # 166
    parser.add_argument('--model_path',default='model/mlp_39.pt')
    parser.add_argument('--best_model_path',default='new_log/model/mlp.ep680')
    parser.add_argument('--first_model_path',default='new_log/model/mlp.ep546')
    parser.add_argument('--two_model',default=0)

    args = parser.parse_args()

    main(args)