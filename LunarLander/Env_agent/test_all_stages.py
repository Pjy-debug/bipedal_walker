import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import create_dataset_new,create_neg_dataset, create_episode_dataset, create_distill_dataset
from utils.criticality_model import Mlp, Reward_Model, Criticality_model_mlp,Criticality_model, Criticality_model_trans
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
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

    plt.scatter([k * (len(loss_neg)/len(loss_pos)) for k in range(len(loss_pos))], 
            loss_pos,  
            alpha=0.7, 
            marker=markers[0], 
            c=colors[0], 
            label=labels[0])
    
    plt.scatter(range(len(loss_neg)), 
                loss_neg,  
                alpha=0.7, 
                marker=markers[1], 
                c=colors[1], 
                label=labels[1])
    

    
    plt.title('Scores')
    plt.ylabel('Scores'); plt.xlabel('Index')
    plt.savefig('/home/teamcommon/tyy/MyLander/Env_agent/prediction.png')
def test(args,test_loader):
    # test_loader
    # print(args)
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=args.log_dir)
    pre_model1 = Reward_Model()
    pre_model1.load_state_dict(torch.load('/home/teamcommon/tyy/MyLander/Env_agent/stage1/40new_rw_model_f5_best.pt'))
    pre_model1.eval().to(device)
    #修改  暂时只测stage1的model
    pre_model2 = Criticality_model_trans()
    pre_model2.load_state_dict(torch.load('/home/teamcommon/tyy/MyLander/Env_agent/stage2/model/trans_49.pt'))
    pre_model2.eval().to(device)
    
    pre_model3 = Criticality_model_trans()
    pre_model3.load_state_dict(torch.load('/home/teamcommon/tyy/MyLander/Env_agent/stage3/reward0_stage3_final_model.pt'))
    pre_model3.eval().to(device)
    
    
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.MSELoss()
    loss_BCE = torch.nn.BCELoss(reduce=False)
    loss_MSE = torch.nn.MSELoss(reduce=False)

    # Build Trainer
    losses, accs, precisions, recalls, TPRs, FPRs = 0, 0, 0, 0, 0, 0
    n_batches, n_samples = len(test_loader), len(test_loader.dataset)

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    pred_pos = 0
    label_pos = 0
    label_neg = 0
    pred_neg = 0
    # 新增: 存储所有标签和预测结果
    all_labels = []
    all_preds = []
    # 新增修改: 存储正负样本的奖励值，而不是损失值
    reward_pos_list = []
    reward_neg_list = []

    false_pos = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i % 100 == 0:
                print(i)
            inputs, labels = map(lambda x: x.to(device), batch)
            labels.reshape(-1, 2)
            labels_cls = torch.tensor(labels[:, 0], dtype=torch.int64)
            labels_dist = labels[:, 1]
            # inputs: (batch_size, seq_len), |labels| : (batch_size)
            
            
            bs = inputs.size(0)
            #print(bs)
            min_value,_ = torch.min(inputs, dim=1, keepdim=True)
            #print(min_value,min_value.shape)
            max_value,_ = torch.max(inputs, dim=1, keepdim=True)
            #print(min_value.repeat(1,25).shape)
            # inputs = (inputs - min_value.repeat(1,11)) / (max_value - min_value).repeat(1,11)
            #print(inputs.shape)
            
            #outputs1 = pre_model1(inputs)
            #暂时修改
            outputs3, feats3,_,_ = pre_model3(inputs,inputs,0.5)
            #outputs3, feats3,_,_ = pre_model3(cur_inputs,cur_inputs,0.5)
            """
            outputs_pos1, feats_pos1,_,_ = pre_pos1(inputs,inputs,0.5)
            outputs_pos2, feats_pos2,_,_ = pre_pos2(inputs,inputs,0.5)
            outputs3,_,_ = pre_model3(cur_inputs)
            re_outputs = reward_model(cur_inputs)
            re_outputs2 = reward_model2(cur_inputs)
            """
            #rewards = outputs1['rewards']
            #loss_neg = 0.5 * loss_input_neg + 0.5 * loss_hid
            #pred_cls1 = (outputs1['rewards'][:,-1] > 0.2).tolist()
            #pred_cls2 = outputs2[:,-1]> 0.5
            #以下暂时修改
            pred_cls3 = outputs3[:,-1]> 0.5
            """
            pred_cls_pos1 = outputs_pos1[:,-1]>0.5
            pred_cls_pos2 = outputs_pos2[:,-1]>0.5
            pred_cls3 = outputs3[:,-1]> 0.5
            re_rewards = re_outputs["rewards"]
            re_rewards2 = re_outputs2["rewards"]
            #pred_cls = outputs[:,-1]>args.threshold
            """
            
            # outputs: (batch_size, 2)
            # attention_weights: [(batch_size, n_attn_heads, seq_len, seq_len)] * n_layers
            
            """
            acc = (pred_cls == labels_cls).sum()
            # acc = (outputs.argmax(dim=-1) == labels_cls).sum()
            accs += acc.item()
            """
            
            for k in range(len(labels_cls)):
                #暂时修改
                pred_result = pred_cls3[k].item()
                #pred_result = pred_cls1[k]
                """ 
                reward = rewards[k,-1].item()
                if reward < 2.5:
                    pred_result = 0
                else:
                    pred_result = pred_cls2[k].item() 
                """
                    
                
                """
                # 最好
                if not AE_test(loss_neg[k].item()):
                    pred_result = 0
                elif (pred_cls1[k].item() == 0) or (pred_cls2[k].item == 0):
                    # 能否再把正样本检回来
                    #pred_result = 0
                    if (pred_cls_pos1[k].item() == 1) and (pred_cls_pos2[k].item() == 1):
                        pred_result = 1
                    else:
                        pred_result = 0
                else:
                    #pred_result = pred_cls3[k].item()
                    
                    if (re_rewards[k,-1].item() > 2.5) and check(re_rewards2[k,-1].item()):
                        pred_result = pred_cls3[k].item()
                    else:
                        pred_result = 0
                    
                    # pred_result = 1
                """
                """
                if not AE_test(loss_neg[k].item()):
                    pred_result = 0
                elif (pred_cls1[k].item() == 0) and (pred_cls2[k].item == 0):
                    pred_result = 0
                else:
                    #pred_result = outputs3[k][-1].item()
                    pred_result = pred_cls3[k].item()
                """
                # TODO: 修改开始
                # 提取 outputs3 的最后一个值作为奖励值
                sample_reward = outputs3[k,-1].item()
                
                # 根据真实标签，将奖励值存入对应的列表
                if labels_cls[k].item() == 1:
                    reward_pos_list.append(sample_reward)
                else:
                    reward_neg_list.append(sample_reward)
                # TODO: 修改结束
                # 新增: 收集预测结果和真实标签
                all_labels.append(labels_cls[k].item())
                all_preds.append(pred_result)
                    
                
                if pred_result == 1:
                    pred_pos += 1
                    if (labels_cls[k] == 1).item():
                        TP += 1
                        accs += 1
                    else:
                        FP += 1
                        cur_input = inputs[k].squeeze(0).clone().detach().cpu().numpy()
                        false_pos.append(cur_input)
                if pred_result == 0:
                    pred_neg += 1
                    if (labels_cls[k] == 0).item():
                        accs += 1
                        TN += 1
                    else:
                        FN += 1
                
                """
                if (pred_cls1[k].item() == 0) or (pred_cls2[k].item == 0):
                    pred_result = 0
                else:
                    pred_result = pred_cls[k].item()
                
                if pred_result == 1:
                    pred_pos += 1
                    if (labels_cls[k] == 1).item():
                        TP += 1
                        accs += 1
                    else:
                        FP += 1
                if pred_result == 0:
                    pred_neg += 1
                    if (labels_cls[k] == 0).item():
                        accs += 1
                        
                """
                """
                
                # stage1判断，说明为负样本可能性较大
                if (pred_cls1[k].item() == 0) and (pred_cls2[k].item == 0):
                    # 保证不漏掉，应该进一步判断
                    if labels_cls[k].item == 0:
                        accs += 1
                else:
                    # 进行精细化判断，当前在训练的模型
                    if (pred_cls[k] == 1).item() and (labels_cls[k] == 1).item():
                        TP += 1
                        accs += 1
                        # print(TP)
                    if (pred_cls[k] == 1).item() and (labels_cls[k] == 0).item():
                        FP += 1
                    if (pred_cls[k] == 0).item() and (labels_cls[k] == 0).item():
                        accs += 1
               """
            
            # print('TP is',TP)
            #pred_pos += ((pred_cls == 1).sum()).item()
            #pred_neg += ((pred_cls == 0).sum()).item()
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
    # np.save(f'/root/autodl-tmp/data/dataset/stage_distilled_train_neg_all_2.npy',false_pos, allow_pickle=True)
    print(f'TP:{TP},FP:{FP},TN:{TN},FN:{FN},pred_pos:{pred_pos},label_pos:{label_pos},label_neg:{label_neg},TPR:{TPRs},FPR:{FPRs}')
    #新增
    pic(reward_pos_list, reward_neg_list)


    return losses / n_batches, accs / n_samples, precisions, recalls, TPRs, FPRs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input parameters
    #parser.add_argument('--batch_size', default=2048, type=int, help='batch size')
    #GPU显存不足，改一下bs？
    parser.add_argument('--batch_size', default=516, type=int, help='batch size')
    parser.add_argument('--output_model_prefix', default='mlp')

    # Train parameters
    parser.add_argument('--epochs', default=2, type=int, help='the number of epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--final_ratio',default=1,type=int,help='ratio of negative / positive')
    parser.add_argument('--start_ratio',default=10,type=int, help='ratio of negative / positive')
    parser.add_argument('--warm_up_epochs',default=10,type=int)
    parser.add_argument('--threshold', default=0.7)

    # Model parameters
    parser.add_argument('--input_dim', default=11, type=int, help='the number of classes')
    parser.add_argument('--embed_dim', default=256, type=int, help='the number of expected features in the transformer')
    parser.add_argument('--n_layers', default=6, type=int, help='the number of heads in the multi-head attention network')
    parser.add_argument('--n_heads', default=8, type=int, help='the number of multi-head attention heads')
    parser.add_argument('--dropout', default=0.1, type=float, help='the residual dropout value')
    parser.add_argument('--ffn_dim', default=1024, type=int, help='the dimension of the feedforward network')
    parser.add_argument('--num_classes', default=2, type=int, help='the number of classes')

    # path
    parser.add_argument('--dataset_dir',default='/mnt1/brx/dataset/')
    parser.add_argument('--log_dir', default='log/log')
    parser.add_argument('--model_path',default='new_log/model/mlp.ep448')

    args = parser.parse_args()
    
    # build dataset
    """
    test_dataset = create_dataset(args.dataset_dir + 'train_dataset_pos.npy',args.dataset_dir + 'train_dataset_neg_1.npy', 0, 'train')
    """
    """
    dataset_path_pos ='/mnt1/brx/dataset/test_dataset_pos_450.npy'
    dataset_path_neg = '/mnt1/brx/dataset/test_dataset_neg_450.npy'
    test_dataset = create_distill_dataset([dataset_path_pos,dataset_path_neg],'total')
    
    """
    
    dataset_path_pos ='/mnt/mnt1/tyy/data/positive/test_pos.npy'
    #dataset_path_pos ='/mnt1/brx/dataset/test_dataset_pos_episode_450_new.npy'
    #dataset_path_pos = '/root/autodl-tmp/data/distill_dataset/train_dataset_pos_narrow_2.npy'
    test_data_pos = np.load(dataset_path_pos,allow_pickle=True)
    test_data_pos = list(test_data_pos)
    
    random_seed = 42  
    random.seed(random_seed)
    random.shuffle(test_data_pos)  # 随机打乱正样本数据
    train_data_pos = test_data_pos[:len(test_data_pos) // 10]

    dataset_path_neg_1 = '/mnt/mnt1/tyy/data/newnegative/test_neg.npy'
    test_data_neg_1 = np.load(dataset_path_neg_1,allow_pickle=True)
    test_data_neg_1 = list(test_data_neg_1)
    
    """
    dataset_path_neg_2 = args.dataset_dir + 'test_dataset_neg_new_2.npy'
    test_data_neg_2 = np.load(dataset_path_neg_2,allow_pickle=True)
    test_data_neg_2 = list(test_data_neg_2)
    """
    
    test_data_neg = test_data_neg_1 
    
    #test_dataset = create_dataset_new(test_data_pos, test_data_neg,'train')
    test_dataset = create_episode_dataset(test_data_pos, test_data_neg,'total')
    
    # Build DataLoader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    test(args, test_loader)


