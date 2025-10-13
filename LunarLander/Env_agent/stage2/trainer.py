import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.criticality_model import TransformerEncoder,Mlp,Criticality_model_mlp,Criticality_model, Criticality_model_trans
from pathlib import Path
import numpy as np
import math
from torch.optim import lr_scheduler

class Trainer:
    def __init__(self, args, train_pos_loader=None,train_neg_loader1=None, train_neg_loader2=None, test_loader=None):
        self.args = args
        self.train_pos_loader = train_pos_loader
        self.train_neg_loader1 = train_neg_loader1
        self.train_neg_iter1 = iter(self.train_neg_loader1)
        self.train_neg_loader2 = train_neg_loader2
        self.train_neg_iter2 = iter(self.train_neg_loader2)
        self.test_loader = test_loader
        self.device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

        """
        self.model = Mlp(args.input_dim, embed_dim_1=args.embed_dim_1, 
                             embed_dim_2=args.embed_dim_2, num_classes=args.num_classes)
        """
        if not args.is_resume:
            #self.model = Criticality_model_mlp().to(self.device)
            #self.model = Criticality_model().to(self.device)
            self.model = Criticality_model_trans().to(self.device)
        # resume = 1 恢复  
        else:
            self.model = Criticality_model_trans().to(self.device)
            self.model.load_state_dict(torch.load('model/trans_49.pt'))
            self.model.to(self.device)
            """
            for name,param in self.model.named_parameters():
                if 'cls_head' not in name:
                    param.requires_grad=False
            """
        
        """
        if args.two_model:
        #self.model.to(self.device)
            self.pre_model = torch.load(args.first_model_path)
            self.pre_model.to(self.device)
            self.pre_model.eval()
        """
        self.L = args.epochs - args.start_epoch
        self.epsilon = 0.5

        self.optimizer = optim.Adam(self.model.parameters(), args.lr)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-8)
        self.lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=2, eta_min=1e-5)
        
        self.criterion2 = nn.CrossEntropyLoss()
        self.criterion1 = FocalLoss()
        self.SCLoss = SCLoss(device=self.device)
        self.tripleLoss = TripletLoss()
        self.centerloss = CenterLoss()
        #self.criterion2 = nn.MSELoss()
        self.kdLoss = HintLoss()
        
        #self.weight_h = (35/50)
        #self.weight_l = (31/50)
        self.weight_h = (48/50)
        self.weight_l = (35/50)
        
        self.delta_weight = self.weight_h - self.weight_l
    
    def normal(self, inputs):
        # inputs = torch.cat((cur_inputs[:,0:24],cur_inputs[:,-1].reshape(-1,1)),dim=-1)
        bs = inputs.size(0)
        #print(bs)
        min_value,_ = torch.min(inputs, dim=1, keepdim=True)
        #print(min_value,min_value.shape)
        max_value,_ = torch.max(inputs, dim=1, keepdim=True)
        #print(min_value.repeat(1,25).shape)
        inputs = (inputs - min_value.repeat(1,11)) / (max_value - min_value).repeat(1,11)
        return inputs

    def train(self, epoch):
        #weight = 0.5 * math.cos(math.pi * (epoch - self.args.start_epoch) / self.L) + 0.5 + self.epsilon
        #weight = 0
        #weight = ((epoch - self.args.start_epoch) / self.L)**2
        #weight = (31/50)**2
        #weight = (33.5/50)**2
        
        weight = (self.weight_l + self.delta_weight * (epoch - self.args.start_epoch) / self.L)**2

        losses, accs, precisions, recalls = 0, 0, 0, 0
        n_batches = len(self.train_pos_loader)
        #从101改成100
        n_samples = self.args.batch_size * 100  * n_batches
        #self.train_neg_iter = iter(self.train_neg_loader)
        self.model.train()
        TP1 = 0
        FP1 = 0
        TP2 = 0
        FP2 = 0
        pred_pos = 0
        pred_neg = 0
        label_pos1 = 0
        label_neg1 = 0
        label_pos2 = 0
        label_neg2 = 0
        #新增 为了画图
        # === 添加这两个列表来存储所有样本的分数和标签 ===
        all_pos_scores = []
        all_neg_scores = []
        # ================================================
        
        for i, data1 in enumerate(self.train_pos_loader):
            try:
                data2 = next(self.train_neg_iter1)
            except StopIteration:
                self.train_neg_iter1 = iter(self.train_neg_loader1)
                data2 = next(self.train_neg_iter1)
            
            try:
                data3 = next(self.train_neg_iter2)
            except StopIteration:
                self.train_neg_iter2 = iter(self.train_neg_loader2)
                data3 = next(self.train_neg_iter2)
            
            input1, label1 = data1[0],data1[1]
            input2, label2 = data2[0],data2[1]
            input3, label3 = data3[0],data3[1]
            
            #print(cur_input1)
            """
            input1 = self.normal(cur_input1)
            input2 = self.normal(cur_input2)
            input3 = self.normal(cur_input3)
            """
            
            
            # inputs1和inputs2的batchsize保持一致
            inputs1 = torch.cat((input1,input2),dim=0)
            labels1 = torch.cat((label1,label2),dim=0)
            inputs2 = input3
            labels2 = label3
            # 将 input3 与自身拼接，使其批次大小也为 202
            #inputs2 = torch.cat((input3, input3), dim=0) 
            #labels2 = torch.cat((label3, label3), dim=0)
            
            import random
            c = list(zip(inputs1,labels1))              
            random.shuffle(c)               
            inputs1,labels1 = zip(*c) 
            inputs1 = torch.stack(inputs1,dim=0)
            labels1 = torch.stack(labels1,dim=0)
            #print('inputs1:',inputs1.shape)
            #print('labels1:',labels1.shape)
            inputs1 = inputs1.to(self.device)
            labels1 = labels1.to(self.device)
            
            inputs1.reshape(-1,self.args.input_dim)
            labels1.reshape(-1,2)
            #print('labels1 shape:',labels1.shape)
            labels_cls1 = torch.tensor(labels1[:,0],dtype=torch.int64)
            labels_dist1 = labels1[:,1]
            
            inputs2 = inputs2.to(self.device)
            #print('inputs2:',inputs2.shape)
            labels2 = labels2.to(self.device)
            inputs2.reshape(-1,self.args.input_dim)
            labels2.reshape(-1,2)
            labels_cls2 = torch.tensor(labels2[:,0],dtype=torch.int64)
            # 打印 inputs1 和 inputs2 的形状       
            #print('labels2 shape:',labels2.shape)
            """
            if self.args.two_model:
                _, mix_feats, feats1, feats2 = self.pre_model(inputs1,inputs2,weight)
            """
            outputs, feats, feats1, feats2 = self.model(inputs1,inputs2,weight)
            # outputs: (batch_size, 2)
            # attention_weights: [(batch_size, n_attn_heads, seq_len, seq_len)] * n_layers
            
            # *** 修改开始 ***
            # 获取当前 batch_size (bs)
            batch_size_ce = inputs1.shape[0]  # 大小为 202
        
            outputs_ce = outputs[:batch_size_ce]
            outputs_focal = outputs[batch_size_ce:]
            labels_ce = labels_cls1[:batch_size_ce]
            labels_focal = labels_cls2[:]

            # 分别计算 CE Loss 和 Focal Loss
            #把focal换成CE
            cls_loss1 = self.criterion2(outputs_ce, labels_ce)
            cls_loss2 = self.criterion2(outputs_focal, labels_focal)
            # 处理 cls_loss1
            '''
            # --- 修改开始：使用数值稳定的 softplus 函数 ---
            margin = 0.5
            
            chosen_scores_ce = outputs_ce[labels_cls1 == 1, -1]
            rejected_scores_ce = outputs_ce[labels_cls1 == 0, -1]
            chosen_mean_scores_ce = chosen_scores_ce.mean()
            rejected_mean_scores_ce = rejected_scores_ce.mean()
            # 使用 softplus 替代 -log(sigmoid) 以保证数值稳定性
            cls_loss1 = torch.relu(-(chosen_mean_scores_ce - rejected_mean_scores_ce - margin)).mean()
            
            chosen_scores_focal = outputs_focal[labels_cls2 == 1, -1]
            rejected_scores_focal = outputs_focal[labels_cls2 == 0, -1]
            chosen_mean_scores_focal = chosen_scores_focal.mean()
            rejected_mean_scores_focal = rejected_scores_focal.mean()
            cls_loss2 = torch.relu(-(chosen_mean_scores_focal - rejected_mean_scores_focal - margin)).mean()
            '''


            loss = weight * cls_loss1 + (1-weight) * cls_loss2
            # --- 修改结束 ---

            '''
            cls_loss1 = self.criterion2(outputs, labels_cls1)
            cls_loss2 = self.criterion1(outputs, labels_cls2)
            loss = weight * cls_loss1 + (1-weight) * cls_loss2
            '''
            
            losses += loss.item()
            #修改
            # 将所有预测和标签合并
            all_labels = labels_cls2
            pred_cls = outputs[:,-1]> self.args.threshold

            # acc1 = (pred_cls == labels_cls1).sum()
            # acc2 = (pred_cls == labels_cls2).sum()
            #新修改
            # *** 修正准确率和TP/FP计算 ***
            # 将预测结果也进行拆分
            pred_cls_ce = pred_cls[:batch_size_ce]
            pred_cls_focal = pred_cls[batch_size_ce:]
            # 修正 acc1 和 acc2 的计算，使用正确的拆分后的标签
            acc1 = (pred_cls_ce == labels_ce).sum()
            acc2 = (pred_cls_focal == labels_focal).sum()
            accs += weight * acc1.item() + (1-weight) * acc2.item()
            # 修正 TP1/FP1 和 TP2/FP2 的计算
            TP1 += ((pred_cls_ce == 1) & (labels_ce == 1)).sum().item()
            FP1 += ((pred_cls_ce == 1) & (labels_ce == 0)).sum().item()
            TP2 += ((pred_cls_focal == 1) & (labels_focal == 1)).sum().item()
            FP2 += ((pred_cls_focal == 1) & (labels_focal == 0)).sum().item()
            # *** 修正结束 ***

            # 修正pred_pos等统计量
            pred_pos += (pred_cls_ce == 1).sum().item() + (pred_cls_focal == 1).sum().item()
            pred_neg += (pred_cls_ce == 0).sum().item() + (pred_cls_focal == 0).sum().item()
            label_pos1 += (labels_ce == 1).sum().item()
            label_neg1 += (labels_ce == 0).sum().item()
            label_pos2 += (labels_focal == 1).sum().item()
            label_neg2 += (labels_focal == 0).sum().item()
            '''
            for k in range(len(labels_cls1)):
                # print(outputs.argmax(dim=-1)[k],labels_cls[k],(labels_cls[k]==0).item())
                if (pred_cls[k] == 1).item() and (labels_cls1[k] == 1).item():
                    TP1 += 1
                    # print(TP)
                if (pred_cls[k] == 1).item() and (labels_cls2[k] == 1).item():
                    TP2 += 1
                if (pred_cls[k] == 1).item() and (labels_cls1[k] == 0).item():
                    FP1 += 1
                if (pred_cls[k] == 1).item() and (labels_cls2[k] == 0).item():
                    FP2 += 1
            
            # print(TP)
            pred_pos += ((pred_cls == 1).sum()).item()
            pred_neg += ((pred_cls == 0).sum()).item()
            label_pos1 += ((labels_cls1 == 1).sum()).item()
            label_neg1 += ((labels_cls1 == 0).sum()).item()
            label_pos2 += ((labels_cls2 == 1).sum()).item()
            label_neg2 += ((labels_cls2 == 0).sum()).item()
            '''

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            #修改
            # 将分数添加到列表中用于绘图
            scores_np = outputs[:,-1].cpu().detach().numpy()
            #labels_np = all_labels.cpu().detach().numpy()
            labels_np = torch.cat((labels_ce, labels_focal), dim=0).cpu().detach().numpy()
            all_pos_scores.extend(scores_np[labels_np == 1])
            all_neg_scores.extend(scores_np[labels_np == 0])

        
        if not pred_pos:
            precisions = 0.0
        else:
            #precisions = weight * TP1 / pred_pos + (1-weight) * TP2 / pred_pos
            precisions = (TP1+TP2)/pred_pos
        if (not label_pos1) or (not label_pos2) :
            recalls = 0.0
        else:
            #recalls = weight * TP1 / label_pos1 + (1-weight) * TP2 / label_pos2
            recalls = (TP1+TP2)/(label_pos1+label_pos2)
        if (precisions + recalls):
            F1_score = 2 * precisions * recalls / (precisions + recalls)
        else:
            F1_score = 0
    
        print('Train dataset: Epoch : {}\t>\tLoss: {:.4f} / Acc: {:.4f} / F1:{:.4f} / Precison: {:.4f} / Recall:{:.4f}'.format(epoch, losses / n_batches, accs / n_samples, F1_score, precisions, recalls))
        filename = f'/home/teamcommon/tyy/MyLander/Env_agent/stage2/picture/epoch_{epoch}_f5.png'
        # 调用 pic 函数保存图片
        newpic(all_pos_scores, all_neg_scores, filename)
        
        return losses / n_batches, accs / n_samples, precisions, recalls

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
                
                #inputs = self.normal(inputs)
                outputs, mix_feature, feats1, feats2 = self.model(inputs, inputs, 0.5)

                cls_loss = self.criterion1(outputs, labels_cls)
                loss = cls_loss 
                losses += loss.item()
                
                pred_cls = outputs[:,-1]> self.args.threshold
                #取二分之一的pred_cls
                number = pred_cls.shape[0] // 2
                acc = (pred_cls[:number] == labels_cls).sum()
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
                pred_pos += ((pred_cls == 1).sum()).item()//2
                pred_neg += ((pred_cls == 0).sum()).item()//2
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

    def save(self, epoch, model_prefix='trans_model', root='model'):
        path = Path(root) / (model_prefix + '.ep%d' % epoch)
        if not path.parent.exists():
            path.parent.mkdir()
        #torch.save(self.model, path)
        torch.save(self.model.state_dict(),f'model/trans_{epoch}.pt')

    def save_best(self, model_prefix='best_model', root='newmodel'):
        path = Path(root) / (model_prefix + '.ep')
        if not path.parent.exists():
            path.parent.mkdir()
        torch.save(self.model.state_dict(), path)

        
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

        #这步将相似度矩阵的对角线上的值全置, 因为对比损失不需要自己与自己的相似度
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


class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()
 
    def forward(self, f_s, f_t):
        loss = self.crit(f_s, f_t)
        return loss

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T
 
    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

    
def max_min_normalization(data_value):
    """
    Data normalization using max value and min value
    Args:
        data_value: The data to be normalized
    """
    data_shape = data_value.shape
    # print(data_value.shape)
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    
    data_col_min_values = data_value.min(-1)
    data_col_max_values = data_value.max(-1)
    new_data = (data_value - data_col_min_values) / (data_col_max_values - data_col_min_values)
    
    """
    new_data=np.zeros(shape=(data_rows+1,data_cols))
    for i in range(0, data_rows, 1):
        for j in range(0, data_cols, 1):
            data_col_min_values = min(data_value[:,j])
            data_col_max_values = max(data_value[:,j])
            new_data[i][j] = (data_value[i][j] - data_col_min_values) / (data_col_max_values - data_col_min_values)
    """
    return new_data

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import os

import matplotlib.pyplot as plt
import numpy as np
import os

def pic(loss_pos, loss_neg, filename):
    """
    Plots and saves the score distribution of positive and negative samples using a scatter plot.

    Args:
        loss_pos (list or np.array): A list/array of scores for positive samples.
        loss_neg (list or np.array): A list/array of scores for negative samples.
        epoch (int): The current epoch number.
        save_dir (str): The directory where the plot will be saved.
    """
    markers = ['o', '^']
    colors = ['coral', 'dodgerblue']
    labels = ['positive', 'negative']

    plt.figure(figsize=(14, 5))

    plt.title('Scores')
    plt.ylabel('Scores')
    plt.xlabel('Index')

    # 将正负样本分数合并成一个 NumPy 数组
    all_scores = np.concatenate([loss_pos, loss_neg])

    # 如果 all_scores 不为空，则动态设置 y 轴的显示范围
    if len(all_scores) > 0:
        min_score = np.min(all_scores)
        max_score = np.max(all_scores)
        margin = (max_score - min_score) * 0.1
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
    
    plt.savefig(filename)
    plt.close()
    print(f"Distribution plot saved to {filename}")


def picPR(all_scores, all_labels, epoch, save_dir):
    """
    绘制并保存PR曲线。
    """
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    if len(np.unique(all_labels)) < 2:
        print("Warning: PR curve cannot be plotted with only one class present in labels.")
        return

    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    average_precision = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (Epoch {epoch})')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f'pr_curve_val_epoch_{epoch}.png')
    plt.savefig(filename)
    plt.close()
    print(f"PR curve saved to {filename}")

def newpic(all_pos_scores, all_neg_scores, filename):
    """
    绘制并保存分数分布图。
    """
    # === 添加此检查以避免空数据错误 ===
    if not all_pos_scores and not all_neg_scores:
        print("Warning: No data points to plot. Skipping distribution plot.")
        return
    
    # --- 修改开始：对正负样本分数进行排序 ---
    loss_pos = sorted(all_pos_scores)
    loss_neg = sorted(all_neg_scores)
    # --- 修改结束 ---

    plt.figure()
    plt.title('Score Distribution')
    plt.xlabel('Sample Index')
    plt.ylabel('Score')
    
    colors = ['blue', 'red']  # 正样本蓝色，负样本红色
    markers = ['o', 'x']  # 正样本圆圈，负样本叉叉
    labels = ['Positive Samples', 'Negative Samples']

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

    # 自动调整y轴范围
    combined_scores = loss_pos + loss_neg
    if combined_scores:
        min_score = min(combined_scores)
        max_score = max(combined_scores)
        margin = (max_score - min_score) * 0.1
        plt.ylim(min_score - margin, max_score + margin)

    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Distribution plot saved to {filename}")