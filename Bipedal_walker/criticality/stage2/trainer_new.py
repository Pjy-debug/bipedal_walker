'''*************************************************************************
【文件名】                 （必需）
【功能模块和目的】         存储Trainer类，利用了TransformerEncoder类
【开发者及日期】           （必需）
【更改记录】               （若修改过则必需注明）
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
from criticality_.criticality_model import TransformerEncoder,Mlp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import math
from torch.optim import lr_scheduler

class Trainer:
    def __init__(self, args, train_pos_loader=None,train_neg_loader=None, test_loader=None):
        self.args = args
        self.train_pos_loader = train_pos_loader
        self.train_neg_loader = train_neg_loader
        self.train_neg_iter = iter(self.train_neg_loader)
        self.test_loader = test_loader
        self.device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

        """
        self.model = Mlp(args.input_dim, embed_dim_1=args.embed_dim_1, 
                             embed_dim_2=args.embed_dim_2, num_classes=args.num_classes)
        """
        if not args.is_resume:
            self.model = TransformerEncoder(input_dim=args.input_dim,
                                            seq_len=args.max_seq_len,
                                            n_layers=args.n_layers,
                                            n_heads=args.n_heads,
                                            embed_dim=args.embed_dim,
                                            ff_dim=args.ffn_dim,
                                            dropout=args.dropout,
                                            num_classes=args.num_classes
                                            )
            
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
        
        self.criterion1 = nn.CrossEntropyLoss()
        #self.criterion1 = FocalLoss()
        self.SCLoss = SCLoss(device=self.device)
        self.tripleLoss = TripletLoss()
        self.centerloss = CenterLoss()
        self.criterion2 = nn.MSELoss()
    

    def train(self, epoch):
        weight = 0.5 * math.cos(math.pi * (epoch - self.args.start_epoch) / self.L) + 0.5 + self.epsilon
        losses, accs, precisions, recalls, TPRs, FPRs = 0, 0, 0, 0, 0, 0
        cls_losses, sc_losses = 0, 0
        n_batches, n_samples = len(self.train_pos_loader),len(self.train_pos_loader.dataset)*2
        #self.train_neg_iter = iter(self.train_neg_loader)
        self.model.train()
        TP = 0
        FP = 0
        pred_pos = 0
        label_pos = 0
        label_neg = 0
        pred_neg = 0
        for i, data1 in enumerate(self.train_pos_loader):
            try:
                data2 = next(self.train_neg_iter)
            except StopIteration:
                self.train_neg_iter = iter(self.train_neg_loader)
                data2 = next(self.train_neg_iter)
            
            input1, label1 = data1[0],data1[1]
            input2, label2 = data2[0],data2[1]
            inputs = torch.cat((input1,input2),dim=0)
            labels = torch.cat((label1,label2),dim=0)
            
            import random
            c = list(zip(inputs,labels))              
            random.shuffle(c)               
            inputs,labels = zip(*c) 
            inputs = torch.stack(inputs,dim=0)
            labels = torch.stack(labels,dim=0)
            #print('inputs:',inputs.shape)
            #print('labels:',labels.shape)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            inputs.reshape(-1,self.args.input_dim)
            labels.reshape(-1,2)
            # print(labels.shape)
            labels_cls = torch.tensor(labels[:,0],dtype=torch.int64)
            # print(labels_cls.shape,labels_cls)
            labels_dist = labels[:,1]
            # print(inputs.shape,labels.shape)

            outputs, feats,_ = self.model(inputs)
            # outputs: (batch_size, 2)
            # attention_weights: [(batch_size, n_attn_heads, seq_len, seq_len)] * n_layers

            cls_loss = self.criterion1(outputs, labels_cls)
            #dist_loss = self.criterion2(fall_dist, labels_dist)
            #sc_loss = self.SCLoss(feats,labels_cls)
            #triple_loss = self.tripleLoss(feats, labels_cls)
            #center_loss = self.centerloss(feats, labels_cls)
            loss = cls_loss 
            #loss = cls_loss + sc_loss * weight
            losses += loss.item()
            cls_losses += cls_loss
            #sc_losses += sc_loss
            
            pred_cls = outputs[:,-1]> self.args.threshold
            #print(pred_cls)
            #print(pred_cls == 1)
            
                
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
            # print(TP)
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

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            if i % 1000 == 0:
                print('Train Epoch {} Iteration {} ({}/{})\tLoss: {:.4f} / Acc: {:.4f} / Precison:{:.4f} / Recall:{:.4f}'.format(epoch,i, i, n_batches, losses / (i+1), accs / ((i+1) * self.args.batch_size * 2) * 100., precision, recall))
                print('cls_loss:',cls_losses/(i+1),'sc_loss:',sc_losses/(i+1))
        
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
    
        print('Train dataset: Epoch : {}\t>\tLoss: {:.4f} / Acc: {:.4f}% / F1:{:.4f} / Precison: {:.4f} / Recall:{:.4f}'.format(epoch, losses / n_batches, accs / n_samples * 100., F1_score,precisions, recalls))
        print('cls_loss:',cls_losses/n_batches,'sc_loss:',sc_losses/n_batches)
        print(f'TP:{TP},FP:{FP},pred_pos:{pred_pos},label_pos:{label_pos},label_neg:{label_neg},pred_pos:{pred_pos},pred_neg:{pred_neg},TPR:{TPRs},FPR:{FPRs}')
        return losses / n_batches,accs / n_samples, precisions, recalls, TPRs, FPRs

    def validate(self, epoch):
        weight = 0.5 * math.cos(math.pi * (epoch - self.args.start_epoch) / self.L) + 0.5 + self.epsilon
        losses, accs, precisions, recalls, TPRs, FPRs = 0, 0, 0, 0, 0, 0
        n_batches, n_samples = len(self.test_loader), len(self.test_loader.dataset)

        self.model.eval()
        TP = 0
        FP = 0
        pred_pos = 0
        label_pos = 0
        label_neg = 0
        pred_neg = 0
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                inputs, labels = map(lambda x: x.to(self.device), batch)
                labels.reshape(-1, 2)
                labels_cls = torch.tensor(labels[:, 0],dtype=torch.int64)
                labels_dist = labels[:, 1]
                # inputs: (batch_size, seq_len), |labels| : (batch_size)

                outputs, feats, _ = self.model(inputs)
                # outputs: (batch_size, 2)
                # attention_weights: [(batch_size, n_attn_heads, seq_len, seq_len)] * n_layers

                cls_loss = self.criterion1(outputs, labels_cls)
                # dist_loss = self.criterion2(fall_dist, labels_dist)
                #sc_loss = self.SCLoss(feats,labels_cls)
                #triple_loss = self.tripleLoss(feats, labels_cls)
                #center_loss = self.centerloss(feats, labels_cls)
                #loss = cls_loss + sc_loss * weight
                loss = cls_loss 
                losses += loss.item()
                
                pred_cls = outputs[:,-1]> self.args.threshold
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
                # print(TP)
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
                if i % 1000 == 0:
                    print('Val Epoch {} Iteration {} ({}/{})\tLoss: {:.4f} / Acc: {:.4f} / Precison:{:.4f} / Recall:{:.4f}'.format(epoch, i, i, n_batches, losses / (i+1), accs / ((i+1) * self.args.batch_size) * 100., precision, recall))
        
        if not pred_pos:
            precisions = 0.0
        else:
            precisions = TP / pred_pos
        if not label_pos:
            recalls = 0.0
        else:
            recalls = TP / label_pos
        TPRs = recall
        if not pred_neg:
            FPRs = 0.0
        else:
            FPRs = FP / pred_neg
        if (precisions + recalls):
            F1_score = 2*precisions*recalls / (precisions + recalls)
        else:
            F1_score = 0
    

        print('Val dataset: Epoch: {}\t>\tLoss: {:.4f} / Acc: {:.4f}% /F1:{:.4f}/ Precison: {:.4f} / Recall:{:.4f}'.format(epoch,losses / n_batches,accs / n_samples * 100.,F1_score,precisions, recalls ))
        print(f'TP:{TP},FP:{FP},pred_pos:{pred_pos},label_pos:{label_pos},label_neg:{label_neg},pred_pos:{pred_pos},pred_neg:{pred_neg},TPR:{TPRs},FPR:{FPRs}')

        return losses / n_batches, accs / n_samples, precisions , recalls , TPRs, FPRs

    def save(self, epoch, model_prefix='model', root='new_log/model_new'):
        path = Path(root) / (model_prefix + '.ep%d' % epoch)
        if not path.parent.exists():
            path.parent.mkdir()
        torch.save(self.model, path)

    def save_best(self, model_prefix='best_model', root='new_log/model_new'):
        path = Path(root) / (model_prefix + '.ep')
        if not path.parent.exists():
            path.parent.mkdir()
        torch.save(self.model, path)

        
class FocalLoss(nn.Module):
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
    
    def __init__(self, alpha=0.1, gamma=2, num_classes = 2, size_average=False):
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

class SCLoss(nn.Module):
    def __init__(self,device,temperature=0.25):
        super(SCLoss, self).__init__()
        self.T = temperature  #温度参数T
        self.device = device
    
    def forward(self,representations,label):
        n = label.shape[0]  # batch

        #这步得到它的相似度矩阵
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        #这步得到它的label矩阵，相同label的位置为1
        mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))
        mask = mask.to(self.device)

        #这步得到它的不同类的矩阵，不同类的位置为1
        mask_no_sim = torch.ones_like(mask) - mask

        #这步产生一个对角线全为0的，其他位置为1的矩阵
        mask_dui_jiao_0 = torch.ones(n ,n) - torch.eye(n, n )
        mask_dui_jiao_0 = mask_dui_jiao_0.to(self.device)

        #这步给相似度矩阵求exp,并且除以温度参数T
        similarity_matrix = torch.exp(similarity_matrix/self.T)

        #这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
        similarity_matrix = similarity_matrix*mask_dui_jiao_0


        #这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
        sim = mask*similarity_matrix


        #用原先的对角线为0的相似度矩阵减去相同类别的相似度矩阵就是不同类别的相似度矩阵
        no_sim = similarity_matrix - sim


        #把不同类别的相似度矩阵按行求和，得到的是对比损失的分母(还差一个与分子相同的那个相似度，后面会加上)
        no_sim_sum = torch.sum(no_sim , dim=1)

        '''
        将上面的矩阵扩展一下，再转置，加到sim（也就是相同标签的矩阵上），然后再把sim矩阵与sim_num矩阵做除法。
        至于为什么这么做，就是因为对比损失的分母存在一个同类别的相似度，就是分子的数据。做了除法之后，就能得到
        每个标签相同的相似度与它不同标签的相似度的值，它们在一个矩阵（loss矩阵）中。
        '''
        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
        sim_sum  = sim + no_sim_sum_expend
        loss = torch.div(sim , sim_sum)
        loss = loss.to(self.device)

        '''
        由于loss矩阵中，存在0数值，那么在求-log的时候会出错。这时候，我们就将loss矩阵里面为0的地方
        全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。
        '''
        loss = mask_no_sim.to(self.device) + loss + torch.eye(n, n).to(self.device)


        #接下来就是算一个批次中的loss了
        loss = -torch.log(loss)  #求-log
        #loss = torch.sum(torch.sum(loss, dim=1) )/(2*n)  #将所有数据都加起来除以2n
        #最后一步也可以写为---建议用这个， (len(torch.nonzero(loss)))表示一个批次中样本对个数的一半
        loss = torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss)))

        #print(loss)  #0.9821
        return loss

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)  # 获得一个简单的距离triplet函数

    def forward(self, inputs, labels):

        n = inputs.size(0)  # 获取batch_size，这里的inputs就是输入矩阵,即batchsize * 特征维度
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n) #每个数平方后，进行加和（通过keepdim保持2维），再扩展成nxn维
        dist = dist + dist.t()  # 这样每个dis[i][j]代表的是第i个特征与第j个特征的平方的和
        dist.addmm_(1, -2, inputs, inputs.t())  # 然后减去2倍的 第i个特征*第j个特征 从而通过完全平方式得到 (a-b)^2
        dist = dist.clamp(min=1e-12).sqrt()  # 然后开方

        # For each anchor, find the hardest positive and negative
        mask = labels.expand(n, n).eq(labels.expand(n, n).t())  # 这里mask[i][j] = 1代表i和j的label相同， =0代表i和j的label不相同
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # 在i与所有有相同label的j的距离中找一个最大的
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # 在i与所有不同label的j的距离找一个最小的
        dist_ap = torch.cat(dist_ap)  # 将list里的tensor拼接成新的tensor
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)  # 声明一个与dist_an相同shape的全1tensor
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=2, feat_dim=256, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
 
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
 
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
 
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
 
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
 
        return loss
 
    def cuda(self, device_id=None):
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))
