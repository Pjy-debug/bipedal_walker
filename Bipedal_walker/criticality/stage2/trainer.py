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

class Trainer:
    def __init__(self, args, train_loader=None,test_loader=None):
        self.args = args
        self.train_loader = train_loader
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
            """
            for name,param in self.model.named_parameters():
                if 'cls_head' not in name:
                    param.requires_grad=False
            """
        else:
            self.model = torch.load(args.model_path)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), args.lr)
        self.criterion1 = nn.CrossEntropyLoss()
        #self.criterion1 = FocalLoss()
        self.tripleLoss = TripletLoss()
        self.centerloss = CenterLoss()
        self.criterion2 = nn.MSELoss()

    def train(self, epoch):
        losses, accs, precisions, recalls, TPRs, FPRs = 0, 0, 0, 0, 0, 0
        n_batches, n_samples = len(self.train_loader), len(self.train_loader.dataset)

        self.model.train()
        TP = 0
        FP = 0
        pred_pos = 0
        label_pos = 0
        label_neg = 0
        pred_neg = 0
        for i, (inputs, labels) in enumerate(self.train_loader):
            # print('inputs:',inputs)
            # print('labels:',labels)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # inputs, labels = map(lambda x: x.to(self.device), batch)
            inputs.reshape(-1,self.args.input_dim)
            labels.reshape(-1,2)
            # print(labels.shape)
            labels_cls = torch.tensor(labels[:,0],dtype=torch.int64)
            # print(labels_cls.shape,labels_cls)
            labels_dist = labels[:,1]
            # print(inputs.shape,labels.shape)

            outputs, feats = self.model(inputs)
            # outputs: (batch_size, 2)
            # attention_weights: [(batch_size, n_attn_heads, seq_len, seq_len)] * n_layers

            cls_loss = self.criterion1(outputs, labels_cls)
            #dist_loss = self.criterion2(fall_dist, labels_dist)
            triple_loss = self.tripleLoss(feats, labels_cls)
            center_loss = self.centerloss(feats, labels_cls)
            loss = cls_loss + triple_loss + center_loss
            losses += loss.item()
            
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

            if i % 1000 == 0:
                print('Train Epoch {} Iteration {} ({}/{})\tLoss: {:.4f} / Acc: {:.4f} / Precison:{:.4f} / Recall:{:.4f}'.format(epoch,
                        i, i, n_batches, losses / (i+1), accs / ((i+1) * self.args.batch_size) * 100., precision, recall))
        
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
        
        print('Train dataset: Epoch : {}\t>\tLoss: {:.4f} / Acc: {:.4f}% /F1:{:.4f}/ Precison: {:.4f} / Recall:{:.4f}'.format(epoch, losses / n_batches,accs / n_samples * 100.,F1_score, precisions, recalls))
        print(f'TP:{TP},FP:{FP},pred_pos:{pred_pos},label_pos:{label_pos},label_neg:{label_neg},pred_pos:{pred_pos},pred_neg:{pred_neg},TPR:{TPRs},FPR:{FPRs}')
        return losses / n_batches,accs / n_samples, precisions, recalls, TPRs, FPRs

    def validate(self, epoch):
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

                outputs, feats = self.model(inputs)
                # outputs: (batch_size, 2)
                # attention_weights: [(batch_size, n_attn_heads, seq_len, seq_len)] * n_layers

                cls_loss = self.criterion1(outputs, labels_cls)
                # dist_loss = self.criterion2(fall_dist, labels_dist)
                triple_loss = self.tripleLoss(feats, labels_cls)
                center_loss = self.centerloss(feats, labels_cls)
                loss = cls_loss + triple_loss + center_loss
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
                if not pred_neg:
                    FPR = 0.0
                else:
                    FPR = FP / pred_neg
                if i % 1000 == 0:
                    print('Val Epoch {} Iteration {} ({}/{})\tLoss: {:.4f} / Acc: {:.4f} / Precison:{:.4f} / Recall:{:.4f}'.format(epoch,
                        i, i, n_batches, losses / (i+1), accs / ((i+1) * self.args.batch_size) * 100., precision, recall))
        
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
    
    def __init__(self, alpha=0.75, gamma=0, num_classes = 2, size_average=False):
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


#torch.manual_seed(42)

