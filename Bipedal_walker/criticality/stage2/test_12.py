import sys
import os
# 获取当前文件的绝对路径
current_path = os.path.abspath(__file__)
# 向上一级目录
parent_dir = os.path.dirname(os.path.dirname(current_path))
# 将criticality目录添加到Python的模块搜索路径
sys.path.append(parent_dir)

# 现在可以导入criticality_目录下的函数了
from data_.data_utils import *
from criticality_.criticality_model import Mlp
from plot.draw_two import draw_roc, draw_pr
from stage1.reward_model import Reward_Model

import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import  matplotlib.pyplot as plt
import tqdm
import torch.nn.functional as F
from statistics import median
import numpy as np

def load_data(dataset_path_pos, dataset_path_neg_root):
    print('pos data is being loaded...')
    data_pos = np.load(dataset_path_pos, allow_pickle=True)

    # for all neg data and neg val data
    print('neg data is being loaded...')
    data_neg=[]
    file_names = os.listdir(dataset_path_neg_root)
    neg_data_list = [
        dataset_path_neg_root + file_name for file_name in file_names if (file_name.endswith('.npy') and not file_name.endswith('all.npy'))]
    for item in tqdm.tqdm(neg_data_list):
        data_neg+=np.load(item, allow_pickle=True).tolist()

    # print(f'len of data pos is {len(data_pos)}, len of data neg is {len(data_neg)}')
    # len of data pos is 587, len of data neg is 14991542   
    
    test_pos_size = round(len(data_pos)/5)
    test_neg_size = round(len(data_neg)/10)
    print(f'test_pos_size is {test_pos_size}')
    print(f'test_neg_size is {test_neg_size}')
    data_pos_train=data_pos[0:test_pos_size]
    data_neg_train=data_neg[0:test_neg_size]
    return [data_pos_train, data_neg_train]


def test_stage1():
        data_pos_train, data_neg_train = load_data('/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data_pos_new/transitions_all.npy', '/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data_neg_new/')
        dataset = create_dataset_new(data_pos_train, data_neg_train, 0, 'test')
        test_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        losses, accs, precisions, recalls, TPRs, FPRs = 0, 0, 0, 0, 0, 0
        cls_losses, sc_losses = 0, 0
        n_batches = len(test_loader)

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

        
        stage2_input=[]
        stage2_label=[]# 其实都是0
        

        for i, data in tqdm.tqdm(enumerate(test_loader)):
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
                        if test_turn == 8:
                            stage2_input.append(cur_input)
                            stage2_label.append(label)
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
        
        return pos_scores, neg_scores, TPR, FPR, precison, stage2_input, stage2_label, TP, FP, TN, FN

def test_stage2(args,stage2_input, stage2_label):
    stage2_dataset = MyDataset(stage2_input, stage2_label)
    test_loader = DataLoader(stage2_dataset, batch_size=args.batch_size, shuffle=True)

    # test_loader
    # print(args)
    writer = SummaryWriter(log_dir=args.log_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = torch.load('/home/teamcommon/pjy/Bipedal_walker/criticality/stage2/storage/model.ep899')
    print(f'model loaded.')
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
    criterion2 = FocalLoss()
    Softmax = torch.nn.Softmax(dim=-1)

    # Build Trainer
    losses, accs, precisions, recalls, TPRs, FPRs = 0, 0, 0, 0, 0, 0
    n_batches, n_samples = len(test_loader), len(test_loader.dataset)
    
    
    ground_pos_list = []
    ground_neg_list = []

    turns=19

    TP        = [0 for _ in range(turns)]
    FP        = [0 for _ in range(turns)]
    TN        = [0 for _ in range(turns)]
    FN        = [0 for _ in range(turns)]
    pred_pos  = [0 for _ in range(turns)]
    label_pos = [0 for _ in range(turns)]
    label_neg = [0 for _ in range(turns)]
    pred_neg  = [0 for _ in range(turns)]

    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(test_loader)):
            inputs, labels = map(lambda x: x.to(device), batch)
            labels_cls = torch.tensor(labels[:], dtype=torch.int64)
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
            #print(outputs.shape)
            bs = len(outputs)//2
            outputs_1=outputs[0:bs, :]
            outputs_2=outputs[bs:, :]
            #print(outputs_1.shape, outputs_2.shape, labels_cls.shape)
            #print(outputs.shape, labels_cls.shape)
            loss_1 = criterion2(outputs_1, labels_cls)
            loss_2 = criterion1(outputs_2, labels_cls)
            # dist_loss = criterion2(fall_dist, labels_dist)
            loss = 0.5 * loss_1 + 0.5 * loss_2
            losses += loss.item()
            
            outputs= 0.5*outputs_1 + 0.5*outputs_2
            pred_cls_raw=outputs[:,-1]
            
            pred_cls_raw_max=1
            pred_cls_raw_min=0
            pred_cls_raw_mid=(pred_cls_raw_min+pred_cls_raw_max)/2

            for k in range(len(labels_cls)):
                for test_turn in range(turns):
                    pred_cls_raw_threshold = -0.999+0.111*test_turn
                    pred_cls_raw_threshold_new = pred_cls_raw_mid+pred_cls_raw_threshold * (pred_cls_raw_max - pred_cls_raw_min) * 0.5
                    
                    if pred_cls_raw[k]>pred_cls_raw_threshold_new:
                        if labels_cls[k]==1:
                            TP[test_turn] += 1
                        else:
                            FP[test_turn] += 1
                    else:
                        if labels_cls[k]==0:
                            TN[test_turn] += 1
                        else:
                            FN[test_turn] += 1
                if labels_cls[k] == 1:
                    ground_pos_list.append(pred_cls_raw[k].cpu().item())
                if labels_cls[k] == 0:
                    ground_neg_list.append(pred_cls_raw[k].cpu().item())
    precison=[]
    recall=[]
    TPR=[]
    FPR=[]

    print(f'TP is {TP}\nFP is {FP}\nTN is {TN}\nFN is {FN}\n')
    TPR = [(tp / (tp + fn) if (tp + fn) != 0 else 1) for tp, fn in zip(TP, FN)] # recall rate
    recall = TPR
    FPR = [fp / (fp + tn) for fp, tn in zip(FP, TN)]
    precison = [(tp / (tp + fp) if (tp + fp) != 0 else 1) for tp, fp in zip(TP, FP)]
    print(f'TPR is {TPR}\nFPR is {FPR}\nprecison is {precison}\n')
        

    # if (precisions + recalls):
    #     F1_score = 2*precisions*recalls / (precisions + recalls)
    # else:
    #     F1_score = 0

    
    return losses / n_batches, precison, recall, TPR, FPR, ground_pos_list, ground_neg_list, TP, FP, TN, FN

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


def pic(loss_pos, loss_neg, save_path):
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
    plt.savefig(save_path)

    print('figure generated')

        
class FocalLoss(torch.nn.Module):
    """
    def __init__(self,weight=None,alpha=0.99,gamma=2):
        super(FocalLoss,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(weight=weight)
    
    def forward(self,input,target):
        logp = self.ce(input,target)
        p = torch.exp(-logp)
        loss = (1-p) ** self.gamma * logp * self.alpha
        return loss
    """
    
    def __init__(self, alpha=0.1, gamma=0, num_classes = 2, size_average=False):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)      
        步骤详细的实现了 focal_loss损失函数.
        :param alpha: 类别权重.当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....]
        :param gamma: 难易样本调节参数
        :param num_classes: 类别数量
        :param size_average: 损失计算方式,默认取均值
        """
        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   
            # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算        
        :param preds:   预测类别 [B,C]  C类别数        
        :param labels:  实际类别 [B]        
        :return:
        """        
        # assert preds.dim()==2 and labels.dim()==1        
        preds = preds.view(-1,preds.size(-1))        
        self.alpha = self.alpha.to(preds.device)        
        preds_softmax = F.softmax(preds, dim=1) 
        preds_logsoft = torch.log(preds_softmax)
        
        # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))           
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))        
        self.alpha = self.alpha.gather(0,labels.view(-1))    
        
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  
        loss = torch.mul(self.alpha, loss.t())        
        if self.size_average:        
            loss = loss.mean()        
        else:            
            loss = loss.sum()        
        return loss

def test():
    pos_scores_stage1, neg_scores_stage1, TPR_stage1, FPR_stage1, precison_stage1, stage2_input, stage2_label, TP_stage1, FP_stage1, TN_stage1, FN_stage1 = test_stage1()
    pic(pos_scores_stage1, neg_scores_stage1, save_path='/home/teamcommon/pjy/Bipedal_walker/criticality/stage2/result/stage1_test_scores.png')
    plt.clf()
    draw_pr(TPR_stage1, precison_stage1, '/home/teamcommon/pjy/Bipedal_walker/criticality/stage2/result/stage1_test_pr_curve.png')
    plt.clf()
    draw_roc(FPR_stage1, TPR_stage1, '/home/teamcommon/pjy/Bipedal_walker/criticality/stage2/result/stage1_test_roc_curve.png')
    plt.clf()
    print('stage1 test done.')
    losses_stage2, precison_stage2, recall_stage2, TPR_stage2, FPR_stage2, ground_pos_list_stage2, ground_neg_list_stage2, TP_stage2, FP_stage2, TN_stage2, FN_stage2 = test_stage2(args,stage2_input, stage2_label)
    pic(ground_pos_list_stage2, ground_neg_list_stage2, '/home/teamcommon/pjy/Bipedal_walker/criticality/stage2/result/stage2_test_scores.png')
    plt.clf()

    TP_all = TP_stage2.copy()
    FP_all = FP_stage2.copy()
    TN_all = TN_stage2.copy()
    for i in range(len(TN_all)):
        TN_all[i] += TN_stage1[8]
    FN_all = FN_stage2.copy()
    for i in range(len(FN_all)):
        FN_all[i] += FN_stage1[8]

    TPR_all = [(tp / (tp + fn) if (tp + fn) != 0 else 1) for tp, fn in zip(TP_all, FN_all)] # recall rate
    recall_all = TPR_all
    FPR_all = [fp / (fp + tn) for fp, tn in zip(FP_all, TN_all)]
    precison_all = [(tp / (tp + fp) if (tp + fp) != 0 else 1) for tp, fp in zip(TP_all, FP_all)]
    draw_pr(TPR_all, precison_all, '/home/teamcommon/pjy/Bipedal_walker/criticality/stage2/result/stage1and2_test_pr_curve.png')
    plt.clf()
    draw_roc(FPR_all, TPR_all, '/home/teamcommon/pjy/Bipedal_walker/criticality/stage2/result/stage1and2_test_roc_curve.png')
    plt.clf()
    print('stage2 test done.')

def my_a_test():
    TP_stage1=[156, 156, 156, 156, 156, 156, 155, 155, 155, 155, 154, 153, 142, 139, 131, 122, 113, 94, 62]
    FP_stage1=[1493419, 1067881, 835149, 709888, 645939, 568521, 490969, 432801, 357814, 286918, 217023, 159236, 113933, 78776, 59048, 40532, 25452, 11758, 1488]
    TN_stage1=[6206, 431744, 664476, 789737, 853686, 931104, 1008656, 1066824, 1141811, 1212707, 1282602, 1340389, 1385692, 1420849, 1440577, 1459093, 1474173, 1487867, 1498137]
    FN_stage1=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 14, 17, 25, 34, 43, 62, 94]
    TPR_stage1=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9935897435897436, 0.9935897435897436, 0.9935897435897436, 0.9935897435897436, 0.9871794871794872, 0.9807692307692307, 0.9102564102564102, 0.8910256410256411, 0.8397435897435898, 0.782051282051282, 0.7243589743589743, 0.6025641025641025, 0.3974358974358974]
    FPR_stage1=[0.9958616320746854, 0.7120986913395015, 0.5569052263065767, 0.4733770109193965, 0.4307336834208552, 0.3791087771942986, 0.3273945152954905, 0.2886061515378845, 0.23860231724597816, 0.1913264982912395, 0.14471817954488622, 0.10618387930315912, 0.0759743269150621, 0.05253046594982079, 0.039375177127615235, 0.027028090355922314, 0.016972243060765193, 0.00784062682337251, 0.0009922480620155039]
    precison_stage1=[0.00010444738295699914, 0.00014606235551764592, 0.00018675813026379585, 0.00021970469435696942, 0.00024145056067606158, 0.000274320923828465, 0.0003156025769459444, 0.00035800404660057836, 0.00043299838812858096, 0.0005399323517014836, 0.0007090990298236001, 0.0009599156779953447, 0.0012447950909489371, 0.0017613888360894633, 0.002213623075753223, 0.0030009347173709845, 0.004420105613142969, 0.007931150860614243, 0.04]
    

    TP_stage2=[155, 155, 155, 155, 154, 153, 153, 151, 151, 151, 151, 151, 150, 148, 148, 0, 0, 0, 0]
    FP_stage2=[357814, 357814, 357814, 357814, 129014, 121835, 119815, 119045, 115819, 114709, 113336, 111041, 109818, 106487, 101720, 0, 0, 0, 0]
    TN_stage2=[0, 0, 0, 0, 228800, 235979, 237999, 238769, 241995, 243105, 244478, 246773, 247996, 251327, 256094, 357814, 357814, 357814, 357814]
    FN_stage2=[0, 0, 0, 0, 1, 2, 2, 4, 4, 4, 4, 4, 5, 7, 7, 155, 155, 155, 155]
    TPR_stage2 = [(tp / (tp + fn) if (tp + fn) != 0 else 1) for tp, fn in zip(TP_stage2, FN_stage2)] # recall rate
    recall_stage2 = TPR_stage2
    FPR_stage2 = [fp / (fp + tn) for fp, tn in zip(FP_stage2, TN_stage2)]
    precison_stage2 = [(tp / (tp + fp) if (tp + fp) != 0 else 1) for tp, fp in zip(TP_stage2, FP_stage2)]
    print(f'for stage2, TPR is {TPR_stage2}\nFPR is {FPR_stage2}\nprecison is {precison_stage2}\n')
    draw_pr(TPR_stage2, precison_stage2, '/home/teamcommon/pjy/Bipedal_walker/criticality/stage2/result_another/stage2_only_test_pr_curve.png')
    plt.clf()
    draw_roc(FPR_stage2, TPR_stage2, '/home/teamcommon/pjy/Bipedal_walker/criticality/stage2/result_another/stage2_only_test_roc_curve.png')
    plt.clf()

    TP_all = TP_stage2.copy()
    FP_all = FP_stage2.copy()
    TN_all = TN_stage2.copy()
    for i in range(len(TN_all)):
        TN_all[i] += TN_stage1[8]
    FN_all = FN_stage2.copy()
    for i in range(len(FN_all)):
        FN_all[i] += FN_stage1[8]
    
    print(f'TP is {TP_all}\nFP is {FP_all}\nTN is {TN_all}\nFN is {FN_all}\n')

    TPR_all = [(tp / (tp + fn) if (tp + fn) != 0 else 1) for tp, fn in zip(TP_all, FN_all)] # recall rate
    recall_all = TPR_all
    FPR_all = [fp / (fp + tn) for fp, tn in zip(FP_all, TN_all)]
    precison_all=[]
    for tp, fp in zip(TP_all, FP_all):
        if (tp + fp) != 0:
            precison_all.append(tp / (tp + fp))
    TPR_all_2=TPR_all[0:len(precison_all)]
    draw_pr(TPR_all_2, precison_all, '/home/teamcommon/pjy/Bipedal_walker/criticality/stage2/result_another/stage1and2_test_pr_curve.png')
    draw_pr(TPR_stage1, precison_stage1, '/home/teamcommon/pjy/Bipedal_walker/criticality/stage2/result_another/stage1_test_pr_curve.png')
    plt.clf()
    draw_roc(FPR_all, TPR_all, '/home/teamcommon/pjy/Bipedal_walker/criticality/stage2/result_another/stage1and2_test_roc_curve.png')
    draw_roc(FPR_stage1, TPR_stage1, '/home/teamcommon/pjy/Bipedal_walker/criticality/stage2/result_another/stage1_test_roc_curve.png')
    plt.clf()
    print('stage2 test done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input parameters
    # 2048
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--output_model_prefix', default='mlp')

    # Train parameters
    parser.add_argument('--start_epoch', default=801, type=int, help='the number of epochs')
    parser.add_argument('--epochs', default=900, type=int, help='the number of epochs')
    # 1e-4
    parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
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
    # parser.add_argument('--dataset_dir',default='/home/yjx/tta_new/data/dataset/')
    parser.add_argument('--dataset_dir',default='/mnt1/hyj/Acc_Test/tta_new/data/test_2025-6-8/processed_data/')
    parser.add_argument('--log_dir', default='../new_log/log')
    parser.add_argument('--is_resume',default=0)
    parser.add_argument('--is_chuan',default=0)
    # 619
    # 865,868,880,887
    # 1087: 提炼过的平衡数据集上训练出来的
    # 113 116 127
    # 负样本特征学习已经到达39
    # 166
    parser.add_argument('--model_path',default='storage/mlp.ep546')
    parser.add_argument('--best_model_path',default='storage/mlp.ep680')
    parser.add_argument('--first_model_path',default='storage/mlp.ep546')
    parser.add_argument('--two_model',default=1)

    args = parser.parse_args()

    # test_12
    # test()

    my_a_test()