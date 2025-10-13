import sys
import os
# 获取当前文件的绝对路径
current_path = os.path.abspath(__file__)
# 向上一级目录
parent_dir = os.path.dirname(os.path.dirname(current_path))
# 将criticality目录添加到Python的模块搜索路径
sys.path.append(parent_dir)

# 现在可以导入criticality_目录下的函数了
from data_.data_utils import create_dataset
from criticality_.criticality_model import Mlp

import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import  matplotlib.pyplot as plt



def test(args,test_loader):
    # test_loader
    # print(args)
    writer = SummaryWriter(log_dir=args.log_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = torch.load(args.model_path)
    model.eval()
    model.to(device)
    '''
    if args.two_model:
        cls_model = torch.load(args.first_model_path)
        cls_model.eval()
        cls_model.to(device)
    '''
    
    """
    # build dataset
    test_dataset = create_dataset(args.dataset_dir + 'test_dataset_pos.npy',args.dataset_dir + 'test_dataset_neg.npy', 0, 'test')
    print(f'Successfully build dataset! Length of test dataset is {len(test_dataset)}')

    # Build DataLoader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    """
   
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.MSELoss()
    Softmax = torch.nn.Softmax(dim=-1)

    # Build Trainer
    losses, accs, precisions, recalls, TPRs, FPRs = 0, 0, 0, 0, 0, 0
    n_batches, n_samples = len(test_loader), len(test_loader.dataset)
    TP = 0
    FP = 0
    pred_pos = 0
    label_pos = 0
    label_neg = 0
    pred_neg = 0
    
    ground_pos_list = []
    ground_neg_list = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs, labels = map(lambda x: x.to(device), batch)
            labels.reshape(-1, 2)
            labels_cls = torch.tensor(labels[:, 0], dtype=torch.int64)
            labels_dist = labels[:, 1]
            # inputs: (batch_size, seq_len), |labels| : (batch_size)
            """
            if args.two_model:
                _, mix_feature, feats1, feats2 = model(inputs, inputs, 0.5)
                outputs = Softmax(cls_model.classifier(mix_feature))
            else:
                outputs, mix_feature, feats1, feats2 = model(inputs, inputs, 0.5)
            """
            outputs, mix_feature, feats1, feats2 = model(inputs, inputs, 0.5)
            # outputs: (batch_size, 2)
            # attention_weights: [(batch_size, n_attn_heads, seq_len, seq_len)] * n_layers
            

            cls_loss = criterion1(outputs, labels_cls)
            # dist_loss = criterion2(fall_dist, labels_dist)
            loss = cls_loss
            losses += loss.item()
            
            pred_cls_raw=outputs[:,-1]
            pred_cls = outputs[:,-1]>args.threshold
            acc = (pred_cls == labels_cls).sum()
            # acc = (outputs.argmax(dim=-1) == labels_cls).sum()
            accs += acc.item()
            
            for k in range(len(labels_cls)):
                # print(outputs.argmax(dim=-1)[k],labels_cls[k],(labels_cls[k]==0).item())
                if (pred_cls[k] == 1).item() and (labels_cls[k] == 1).item():
                    TP += 1
                    # print(TP)
                if (pred_cls[k] == 1).item() and (labels_cls[k] == 0).item():
                    FP += 1
                if labels_cls[k] == 1:
                    ground_pos_list.append(pred_cls_raw[k].cpu().item())
                if labels_cls[k] == 0:
                    ground_neg_list.append(pred_cls_raw[k].cpu().item())

            # print('TP is',TP)
            pred_pos += ((pred_cls == 1).sum()).item()
            pred_neg += ((pred_cls == 0).sum()).item()
            label_pos += ((labels_cls == 1).sum()).item()
            label_neg += ((labels_cls == 0).sum()).item()
            
            if not pred_pos:
                precision = 0.0
            else:
                precision = TP / pred_pos
            if not label_pos:
                recall = 0.0
                TPR = 0.0
            else:
                recall = TP / label_pos
                TPR = recall
            if not label_neg:
                FPR = 0.0
            else:
                FPR = FP / label_neg
    
    if not pred_pos:
        precisions = 0.0
    else:
        precisions = TP / pred_pos
    if not label_pos:
        recalls = 0.0
    else:
        recalls = TP / label_pos
    TPRs = recall
    if not label_neg:
        FPRs = 0.0
    else:
        FPRs = FP / label_neg
    if (precisions + recalls):
        F1_score = 2*precisions*recalls / (precisions + recalls)
    else:
        F1_score = 0
    

    print('Test dataset >\tLoss: {:.4f} / Acc: {:.4f}% / F1: {:.4f} / Precison: {:.4f} / Recall:{:.4f} '.format(  losses / n_batches,
                                                                                                    accs / n_samples * 100.,
                                                                                                    F1_score,
                                                                                                    precisions,
                                                                                                    recalls))
    
    print(f'TP:{TP},FP:{FP},pred_pos:{pred_pos},label_pos:{label_pos},label_neg:{label_neg},TPR:{TPRs},FPR:{FPRs}')
    pic(ground_pos_list, ground_neg_list)

    return losses / n_batches, accs / n_samples, precisions, recalls, TPRs, FPRs

    """
    if epoch % 5 == 0:
            trainer.save(epoch, args.output_model_prefix)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        writer.add_scalar('Precision/train', train_precision, epoch)
        writer.add_scalar('Precision/test', test_precision, epoch)
        writer.add_scalar('Recall/train', train_recall, epoch)
        writer.add_scalar('Recall/test', test_recall, epoch)

    writer.close()
    """


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
    plt.savefig('/home/teamcommon/pjy/Bipedal_walker/criticality/stage2/storage/reward_model_test.png')

    print('figure generated')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input parameters
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--output_model_prefix', default='mlp')

    # Train parameters
    parser.add_argument('--start_epoch', default=621, type=int, help='the number of epochs')
    parser.add_argument('--epochs', default=670, type=int, help='the number of epochs')
    # 1e-4
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--final_ratio',default=0,type=int,help='ratio of negative / positive')
    parser.add_argument('--start_ratio',default=0,type=int, help='ratio of negative / positive')
    parser.add_argument('--warm_up_epochs',default=20,type=int)
    # 0.546
    parser.add_argument('--threshold', default=0.5)

    # Model parameters
    parser.add_argument('--input_dim', default=107, type=int, help='the number of classes')
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
    parser.add_argument('--dataset_dir',default='/root/autodl-tmp/data/dataset/')
    parser.add_argument('--log_dir', default='new_log/log')
    parser.add_argument('--is_resume',default=1)
    parser.add_argument('--is_chuan',default=0)
    # 619
    # 865,868,880,887
    # 1087: 提炼过的平衡数据集上训练出来的
    # 113 116 127
    # 负样本特征学习已经到达39
    # 166
    parser.add_argument('--model_path',default='new_log/model_new/mlp.ep668')
    parser.add_argument('--best_model_path',default='new_log/model/mlp.ep680')
    parser.add_argument('--first_model_path',default='new_log/model_new/mlp.ep546')
    parser.add_argument('--two_model',default=1)

    args = parser.parse_args()

    test(args)