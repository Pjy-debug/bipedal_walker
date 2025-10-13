import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, input_dim,embed_dim):
        super(Embedding,self).__init__()
        self.fc = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.fc(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, q, k, v):
        # Q:[bs,n_heads,len_q,d_k]
        # [bs, n_heads, len_q, len_q]
        scores = torch.matmul(q,k.transpose(-1,-2)) / np.sqrt(self.d_k)
        #scores.masked_fill(attn_mask,-1e9)
        attn_weights = nn.Softmax(dim=-1)(scores)
        output = torch.matmul(attn_weights,v)
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = embed_dim // n_heads
        self.d_v = embed_dim // n_heads
        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.scaled_attn = ScaledDotProductAttention(self.d_k)
        self.linear = nn.Linear(n_heads * self.d_v, embed_dim)

    def forward(self, Q, K, V):
        # Q: [bs, len_q, embed_dim]
        # attn_mask: [bs, len_q, len_k]
        bs = Q.size(0)

        q_s = self.W_Q(Q).view(bs,-1,self.n_heads,self.d_k).transpose(1,2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        # [bs, n_heads, len_q, d_v]
        # [bs, n_heads, len_q, len_k]
        attn, attn_weights = self.scaled_attn(q_s, k_s, v_s)
        # [bs, q_len, n_heads * d_v]
        attn = attn.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        # [bs,q_len,embed_dim]
        output = self.linear(attn)
        return output, attn_weights

class FFN(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super(FFN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, ff_dim, bias = False),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim, bias=False))

    def forward(self, inputs):
        output = self.fc(inputs)
        return output

class EncoderLayer(nn.Module):
    def __init__(self,embed_dim, dropout, n_heads,ff_dim):
        super(EncoderLayer,self).__init__()
        self.mltiheadAttn = MultiHeadAttention(embed_dim,n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(embed_dim)

        self.ffn = FFN(embed_dim, ff_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, inputs):
        attn_outputs, attn_weights = self.mltiheadAttn(inputs,inputs,inputs)
        attn_outputs = self.dropout1(attn_outputs)
        attn_outputs = self.layernorm1(inputs+attn_outputs)

        ffn_outputs = self.ffn(attn_outputs)
        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.layernorm2(attn_outputs + ffn_outputs)

        return ffn_outputs, attn_weights

class TransformerEncoder(nn.Module):
    # input_dim = 105
    # state_embedding = 25
    def __init__(self,input_dim=107, seq_len=11, n_layers=3, n_heads=8,
                 embed_dim=256, ff_dim=1024, dropout=0.1, num_classes=2):
        super(TransformerEncoder,self).__init__()
        self.seq_embedding = Embedding(8, embed_dim)
        self.state_embedding = nn.Linear(27,embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1,seq_len,embed_dim))

        self.encoder_layer = EncoderLayer(embed_dim, dropout, n_heads,ff_dim)
        self.layers = nn.ModuleList([self.encoder_layer for _ in range(n_layers)])
        
        """
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim,num_classes)
        )
        """
        #self.layernorm = nn.LayerNorm(embed_dim)
        #self.cls_head = Mlp(input_dim=embed_dim)
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024,embed_dim))
        self.cls_head = nn.Linear(embed_dim,num_classes)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        
        self.init_weights()

    def forward(self, inputs):
        """
        bs, src_len, _ = enc_inputs.shape
        cls_tokens = self.cls_token.repeat(bs,1,1)
        enc_inputs = torch.cat((cls_tokens,x),dim=1)
        """
        input_seq = inputs[:,26:106]
        input_seqs = []
        for k in range(0,80,8):
            input_seqs.append(input_seq[:,k:k+8])
        input_seqs = torch.stack(input_seqs,dim=1)
        # print('shape of input_seqs is',input_seqs.shape)
        input_state = torch.cat((inputs[:,0:26],inputs[:,-1].reshape(-1,1)),dim=-1)
        input_state = input_state.unsqueeze(1)
        # print('shape of input_state is',input_state.shape)
            
        outputs = self.seq_embedding(input_seqs) 
        #print(input_state.shape)
        #print(self.state_embedding)
        outputs = torch.cat((self.state_embedding(input_state),outputs),dim=1)
        outputs += self.pos_embedding
        # print('shape of input of encoder is',outputs.shape)
        # attn_mask = self.get_attn_pad_mask(inputs, inputs)
        attention_weights = []
        for layer in self.layers:
            outputs, attn_weights = layer(outputs)
            attention_weights.append(attn_weights)
        # outputs, _ = torch.max(outputs,dim=1)
        outputs = outputs[:,-1]
        feats1 = outputs
        outputs = self.fc(outputs)
        feats2 = F.normalize(outputs, p=2, dim=-1)
        # [bs,2]
        outputs = self.softmax(self.cls_head(feats2))
        #outputs = self.sigmoid(self.cls_head(outputs))
        #outputs = self.cls_head(outputs)

        return outputs, feats1, feats2

    def get_attn_pad_mask(self,seq_q, seq_k):
        bs, len_q = seq_q.size()
        bs, len_k = seq_k.size()
        # [bs, 1, len_k]
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
        return pad_attn_mask.expand(bs, len_q, len_k)
    
    def init_weights(self):
        """Initialize the weights"""
        self.apply(self._init_weights)
    
    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.1)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def apply(self, fn):
        for module in self.children():
            module.apply(fn)
        fn(self)

class BackBone(nn.Module):
    # input_dim = 105
    # state_embedding = 25
    def __init__(self,input_dim=121, seq_len=11, n_layers=6, n_heads=8,
                 embed_dim=256, ff_dim=1024, dropout=0.1, num_classes=2):
        super(BackBone,self).__init__()
        self.seq_embedding = Embedding(8, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1,seq_len,embed_dim))

        self.encoder_layer = EncoderLayer(embed_dim, dropout, n_heads,ff_dim)
        self.layers = nn.ModuleList([self.encoder_layer for _ in range(n_layers)])
        
        """
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim,num_classes)
        )
        """
        #self.layernorm = nn.LayerNorm(embed_dim)
        #self.cls_head = Mlp(input_dim=embed_dim)
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024,embed_dim))
        
        self.init_weights()

    def forward(self, inputs):
        """
        bs, src_len, _ = enc_inputs.shape
        cls_tokens = self.cls_token.repeat(bs,1,1)
        enc_inputs = torch.cat((cls_tokens,x),dim=1)
        """
        input_seqs = []
        for k in range(0,121,11):
            #input_seqs.append(input_seq[:,k:k+11]) 原来写的有问题？没有input_seq
            input_seqs.append(inputs[:,k:k+11])
        input_seqs = torch.stack(input_seqs,dim=1)
        # print('shape of input_seqs is',input_seqs.shape)
        # print('shape of input_state is',input_state.shape)
            
        outputs = self.seq_embedding(input_seqs) 
        #print(input_state.shape)
        #print(self.state_embedding)
        outputs += self.pos_embedding
        # print('shape of input of encoder is',outputs.shape)
        # attn_mask = self.get_attn_pad_mask(inputs, inputs)
        attention_weights = []
        for layer in self.layers:
            outputs, attn_weights = layer(outputs)
            attention_weights.append(attn_weights)
        # outputs, _ = torch.max(outputs,dim=1)
        outputs = outputs[:,-1]
        feats1 = outputs
        outputs = self.fc(outputs)
        feats2 = F.normalize(outputs, p=2, dim=-1)

        return feats2

    def get_attn_pad_mask(self,seq_q, seq_k):
        bs, len_q = seq_q.size()
        bs, len_k = seq_k.size()
        # [bs, 1, len_k]
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
        return pad_attn_mask.expand(bs, len_q, len_k)
    
    def init_weights(self):
        """Initialize the weights"""
        self.apply(self._init_weights)
    
    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.1)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def apply(self, fn):
        for module in self.children():
            module.apply(fn)
        fn(self)

class FCNorm(nn.Module):
    def __init__(self, num_features, num_classes):
        super(FCNorm, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.linear(F.normalize(x), F.normalize(self.weight))
        return out 

class Classifier(nn.Module):
    # input_dim = 105
    # state_embedding = 25
    def __init__(self,embed_dim=256, num_classes=2):
        super(Classifier,self).__init__()
        self.cls_head = nn.Linear(embed_dim,num_classes)
        
        self.init_weights()

    def forward(self, inputs):
        outputs = self.cls_head(inputs)
        return outputs
    
    def init_weights(self):
        """Initialize the weights"""
        self.apply(self._init_weights)
    
    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
            
    def apply(self, fn):
        for module in self.children():
            module.apply(fn)
        fn(self)

class Criticality_model(nn.Module):
    def __init__(self,input_dim=121, seq_len=11, n_layers=6, n_heads=8,
                 embed_dim=256, ff_dim=1024, dropout=0.1, num_classes=2):
        super(Criticality_model,self).__init__()
        self.backbone1 = BackBone(input_dim, seq_len, n_layers, n_heads,embed_dim, ff_dim, dropout, num_classes)
        self.backbone2 = BackBone(input_dim, seq_len, n_layers, n_heads,embed_dim, ff_dim, dropout, num_classes)
        self.classifier = FCNorm(embed_dim * 2, num_classes)
        self.softmax = nn.Softmax(dim=-1)
        

    def forward(self, inputs1,inputs2,weight,input_mixed_feature=None):
        if input_mixed_feature:
            mixed_feature = input_mixed_feature
        else:
            feats1 = self.backbone1(inputs1)
            feats2 = self.backbone1(inputs2)
            mixed_feature = 2 * torch.cat((weight * feats1, (1-weight) * feats2), dim=-1)
        outputs = self.softmax(self.classifier(mixed_feature))

        return outputs, mixed_feature, feats1, feats2

class Criticality_model_mlp(nn.Module):
    def __init__(self,input_dim=11, embed_dim_1=256, embed_dim_2=1024, num_classes=2):
        super(Criticality_model_mlp,self).__init__()
        self.backbone1 = nn.Sequential(
            nn.Linear(input_dim, embed_dim_1),
            nn.ReLU(),
            nn.Linear(embed_dim_1, embed_dim_2),
            nn.ReLU(),
            nn.Linear(embed_dim_2, embed_dim_2),
            nn.ReLU(),
            nn.Linear(embed_dim_2,embed_dim_1))
        self.backbone2 = nn.Sequential(
            nn.Linear(input_dim, embed_dim_1),
            nn.ReLU(),
            nn.Linear(embed_dim_1, embed_dim_2),
            nn.ReLU(),
            nn.Linear(embed_dim_2, embed_dim_2),
            nn.ReLU(),
            nn.Linear(embed_dim_2,embed_dim_1))
        self.classifier = FCNorm(embed_dim_1 * 2, num_classes)
        self.softmax = nn.Softmax(dim=-1)
        

    def forward(self, inputs1,inputs2,weight,input_mixed_feature=None):
        if input_mixed_feature:
            mixed_feature = input_mixed_feature
        else:
            feats1 = self.backbone1(inputs1)
            #feats2 = self.backbone1(inputs2) ???是不是笔误
            feats2 = self.backbone2(inputs2)
            mixed_feature = 2 * torch.cat((weight * feats1, (1-weight) * feats2), dim=-1)
        outputs = self.softmax(self.classifier(mixed_feature))

        return outputs, mixed_feature, feats1, feats2

class Mlp(nn.Module):
    def __init__(self, input_dim=25, embed_dim_1=512, embed_dim_2=1024, num_classes=2):
        super(Mlp, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, embed_dim_1),
            nn.ReLU(),
            nn.Linear(embed_dim_1, embed_dim_2),
            nn.ReLU(),
            nn.Linear(embed_dim_2, embed_dim_2),
            nn.ReLU(),
            nn.Linear(embed_dim_2,embed_dim_1))
        self.cls_head = nn.Linear(embed_dim_1, num_classes)
        #self.dist_head = nn.Linear(embed_dim_1, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        output = self.cls_head(self.fc(inputs))
        outputs = self.softmax(output)
        #dist = self.dist_head(self.fc(inputs))
        return outputs

class Episode_Encoder(nn.Module):
    # input_dim = 105
    # state_embedding = 25
    def __init__(self,input_dim=11*11, seq_len=11, n_layers=6, n_heads=8,
                 embed_dim=256, ff_dim=1024, dropout=0.1, num_classes=2):
        super(Episode_Encoder,self).__init__()
        self.seq_embedding = Embedding(11, embed_dim)
        # self.state_embedding = nn.Linear(27,embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1,seq_len,embed_dim))

        self.encoder_layer = EncoderLayer(embed_dim, dropout, n_heads,ff_dim)
        self.layers = nn.ModuleList([self.encoder_layer for _ in range(n_layers)])
        
        """
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim,num_classes)
        )
        """
        #self.layernorm = nn.LayerNorm(embed_dim)
        #self.cls_head = Mlp(input_dim=embed_dim)
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024,embed_dim))
        self.cls_head = nn.Linear(embed_dim,num_classes)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        
        self.init_weights()

    def forward(self, inputs):
        """
        bs, src_len, _ = enc_inputs.shape
        cls_tokens = self.cls_token.repeat(bs,1,1)
        enc_inputs = torch.cat((cls_tokens,x),dim=1)
        """
        #print(f'shape of inputs: {inputs.shape}')
        input_seqs= []
        
        #stage3
        '''
        for k in range(0,121,11):
            input_seqs.append(inputs[:,k:k+11])
        input_seqs = torch.stack(input_seqs,dim=1)
        '''
        # print('shape of input_seqs is',input_seqs.shape)
        # print('shape of input_state is',input_state.shape)
        #stage1,2
        input_seqs = inputs
        outputs = self.seq_embedding(input_seqs) 
        #print(input_state.shape)
        #print(self.state_embedding)
        outputs += self.pos_embedding
        # print('shape of input of encoder is',outputs.shape)
        # attn_mask = self.get_attn_pad_mask(inputs, inputs)
        attention_weights = []
        for layer in self.layers:
            outputs, attn_weights = layer(outputs)
            attention_weights.append(attn_weights)
        # outputs, _ = torch.max(outputs,dim=1)
        outputs = outputs[:,-1]
        feats1 = outputs
        # outputs = self.fc(outputs)
        feats2 = F.normalize(outputs, dim=-1)
        # [bs,2]
        #outputs = self.sigmoid(self.cls_head(outputs))
        #outputs = self.cls_head(outputs)

        return feats2

    def get_attn_pad_mask(self,seq_q, seq_k):
        bs, len_q = seq_q.size()
        bs, len_k = seq_k.size()
        # [bs, 1, len_k]
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
        return pad_attn_mask.expand(bs, len_q, len_k)
    
    def init_weights(self):
        """Initialize the weights"""
        self.apply(self._init_weights)
    
    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.1)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def apply(self, fn):
        for module in self.children():
            module.apply(fn)
        fn(self)

class Criticality_model_trans(nn.Module):
    def __init__(self,input_dim=121, seq_len=11, n_layers=3, n_heads=8,
                 embed_dim=256, ff_dim=1024, dropout=0.1, num_classes=2):
        super(Criticality_model_trans,self).__init__()

        #原来6层，改成3层了
        self.backbone1 = Episode_Encoder(input_dim, seq_len, n_layers, n_heads,embed_dim, ff_dim, dropout, num_classes)
        self.backbone2 = Episode_Encoder(input_dim, seq_len, n_layers, n_heads,embed_dim, ff_dim, dropout, num_classes)
        #把embeddim乘以2修改了
        self.classifier = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, num_classes)) # FCNorm(embed_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)
        

    def forward(self, inputs1,inputs2,weight,input_mixed_feature=None):
        if input_mixed_feature:
            mixed_feature = input_mixed_feature
        else:
            feats1 = self.backbone1(inputs1)
            feats2 = self.backbone2(inputs2)
            mixed_feature = torch.cat((feats1, feats2), dim=0)
        outputs = self.softmax(self.classifier(mixed_feature))
        return outputs, mixed_feature, feats1, feats2

class Reward_Model(nn.Module):
    # input_dim = 105
    # state_embedding = 25
    def __init__(self,input_dim=121, seq_len=11, n_layers=6, n_heads=8,
                 embed_dim=256, ff_dim=1024, dropout=0.1, num_classes=2):
        super(Reward_Model,self).__init__()
        self.seq_embedding = Embedding(11, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1,seq_len,embed_dim))

        self.encoder_layer = EncoderLayer(embed_dim, dropout, n_heads,ff_dim)
        self.layers = nn.ModuleList([self.encoder_layer for _ in range(n_layers)])
        
        """
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim,num_classes)
        )
        """
        self.layernorm = nn.LayerNorm(embed_dim)
        #self.cls_head = Mlp(input_dim=embed_dim)
        """
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024,embed_dim))
        self.cls_head = nn.Linear(embed_dim,num_classes)
        """
        self.cls_head = nn.Linear(embed_dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
    
        
        self.init_weights()

    def forward(self, inputs):
        # stage 3
        input_seqs = []
        '''
        for k in range(0,121,11):
            input_seqs.append(inputs[:,k:k+11])
        input_seqs = torch.stack(input_seqs,dim=1)
        '''
        input_seqs = inputs
        # print('shape of input_seqs is',input_seqs.shape)
        #input_state = torch.cat((inputss[:,0:26],inputs[:,-1].reshape(-1,1)),dim=-1)
        # print('shape of input_state is',input_state.shape)
        
            
        outputs = self.seq_embedding(input_seqs) 
        #print(input_state.shape)
        #print(self.state_embedding)
        outputs += self.pos_embedding
        # print('shape of input of encoder is',outputs.shape)
        # attn_mask = self.get_attn_pad_mask(inputs, inputs)
        attention_weights = []
        for layer in self.layers:
            outputs, attn_weights = layer(outputs)
            attention_weights.append(attn_weights)
        # outputs, _ = torch.max(outputs,dim=1)
        # outputs = outputs[:,-1]
        hidden_states = outputs
        rewards = self.sigmoid(self.cls_head(hidden_states)).squeeze(-1) - 0.5
        return {"rewards":rewards}
        
        
        # bs = inputs.shape[0] // 2
        # chosen_as = inputs[:bs]
        # rejected_as = inputs[bs:]
        # chosen_rewards = rewards[:bs]
        # rejected_rewards = rewards[bs:]
        
        # loss = 0
        # chosen_mean_scores = []
        # rejected_mean_scores = []
        # #新增
        # chosen_mean_scores = chosen_rewards[:, -1]
        # rejected_mean_scores = rejected_rewards[:, -1]
        # #pos_loss = -chosen_mean_scores.mean()
        # #neg_loss = rejected_mean_scores.mean()
        # #loss = pos_loss + neg_loss
        # #修改
        # margin = 0.5 # 设置一个期望的最小差距
        # #loss = -torch.log(torch.sigmoid(chosen_mean_scores - rejected_mean_scores - margin)).mean()

        
        # # 损失函数的目标是让 chosen_mean_scores 至少比 rejected_mean_scores 大 margin
        # # 如果 chosen_mean_scores - rejected_mean_scores > margin，则损失为0
        # # 否则，损失为正值，促使模型增大 chosen_mean_scores 或减小 rejected_mean_scores
        # loss = torch.relu(rejected_mean_scores - chosen_mean_scores + margin).mean()
        # '''for i in range(bs):
        #     chosen_a = chosen_as[i]
        #     rejected_a = rejected_as[i]
        #     chosen_reward = chosen_rewards[i]
        #     rejected_reward = rejected_rewards[i]
            
        #     c_truncated_reward = chosen_reward[1:]
        #     r_truncated_reward = rejected_reward[1:]
            
        #     loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
            
        #     chosen_mean_scores.append(chosen_reward[-1]) 
        #     rejected_mean_scores.append(rejected_reward[-1])
            
        # loss = loss / bs
        # chosen_mean_scores = torch.stack(chosen_mean_scores)
        # rejected_mean_scores = torch.stack(rejected_mean_scores)'''
        
        # re = {"loss": loss, "chosen_mean_scores": chosen_mean_scores, "rejected_mean_scores": rejected_mean_scores,
        #      "rewards":rewards}
        # return re
        
        

    def get_attn_pad_mask(self,seq_q, seq_k):
        bs, len_q = seq_q.size()
        bs, len_k = seq_k.size()
        # [bs, 1, len_k]
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
        return pad_attn_mask.expand(bs, len_q, len_k)
    
    def init_weights(self):
        """Initialize the weights"""
        self.apply(self._init_weights)
    
    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.1)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def apply(self, fn):
        for module in self.children():
            module.apply(fn)
        fn(self)


