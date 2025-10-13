import torch
import numpy as np
from torch.utils.data import DataLoader
from data_utils import create_dataset,create_dataset_new,create_dataset_new_2,create_distill_dataset

def test(model_path):
    # test_loader
    # print(args)
    model = torch.load(model_path)
    
    """
    # build dataset
    dataset_path_pos = '/root/autodl-tmp/data/dataset/train_dataset_pos.npy'
    train_data_pos = np.load(dataset_path_pos, allow_pickle=True)
    train_data_pos = list(train_data_pos)
    print(len(train_data_pos))
    
    dataset_path_neg = '/root/autodl-tmp/data/dataset/train_dataset_neg_24.npy' 
    train_data_neg = np.load(dataset_path_neg, allow_pickle=True)
    train_data_neg = list(train_data_neg)
    print(len(train_data_neg))
    
    train_dataset = create_dataset_new(train_data_pos, train_data_neg, 0,'train')
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    print(f'Successfully build dataset! Length of train dataset is {len(train_dataset)}')
    """
    
    dataset_path_pos = '/root/autodl-tmp/data/dataset/distilled_train_dataset_pos.npy'
    train_dataset_pos = create_distill_dataset(dataset_path_pos,'pos')
    print(len(train_dataset_pos))
    
    dataset_path_neg = '/root/autodl-tmp/data/dataset/distilled_train_dataset_neg.npy'
    train_dataset_neg = create_distill_dataset(dataset_path_neg,'neg')
    print(len(train_dataset_neg))
    
    train_dataset = create_distill_dataset([dataset_path_pos, dataset_path_neg],'total')
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    print(f'Successfully build dataset! Length of train dataset is {len(train_dataset)}')
  
    # Build DataLoader
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    model.to(device)
    
    shuffled_inputs_pos = []
    shuffled_inputs_neg = []
    
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i % 100 == 0:
                print(i)
            inputs, labels = map(lambda x: x.to(device), batch)
            labels.reshape(-1, 2)
            labels_cls = torch.tensor(labels[:, 0], dtype=torch.int64)
            labels_dist = labels[:, 1]
            # inputs: (batch_size, seq_len), |labels| : (batch_size)

            outputs, feats,_,_ = model(inputs,inputs,0.5)
            
            pred_cls = outputs[:,-1]> 0.6
            
            for k in range(len(labels_cls)):
                """
                if (labels_cls[k] == 1).item():
                    cur_input = inputs[k].squeeze(0).clone().detach().cpu().numpy()
                    shuffled_inputs_pos.append(cur_input)
                if (labels_cls[k] == 0).item():
                    cur_input = inputs[k].squeeze(0).clone().cpu().detach().cpu().numpy()
                    shuffled_inputs_neg.append(cur_input)
                """
                if (pred_cls[k] == 0).item() and (labels_cls[k] == 1).item():
                    cur_input = inputs[k].squeeze(0).clone().detach().cpu().numpy()
                    shuffled_inputs_pos.append(cur_input)
                if (pred_cls[k] == 1).item() and (labels_cls[k] == 0).item():
                    cur_input = inputs[k].squeeze(0).clone().cpu().detach().cpu().numpy()
                    shuffled_inputs_neg.append(cur_input)
                
                    
    print(len(shuffled_inputs_pos),len(shuffled_inputs_neg))
    np.save(f'/root/autodl-tmp/data/dataset/distilled_train_dataset_pos_303.npy', shuffled_inputs_pos, allow_pickle=True)
    np.save(f'/root/autodl-tmp/data/dataset/distilled_train_dataset_neg_303.npy', shuffled_inputs_neg, allow_pickle=True)
    

test('new_log/model_new/mlp.ep303')
print('finish testing!')