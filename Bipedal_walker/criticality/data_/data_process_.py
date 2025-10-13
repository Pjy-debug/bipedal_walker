'''*************************************************************************
【文件名】                 （必需）
【功能模块和目的】         （必需）
【开发者及日期】           （必需）
【更改记录】               （若修改过则必需注明）
*************************************************************************'''
import numpy as np
import os
import random

# 前人的智慧，注释有点乱
def build_dataset(log_dir,p):
    # log_dir = 'tta/data/processed_data/'
    # p: proportion of train dataset 0.7
    file_names = os.listdir(log_dir)
    # 寻找log_dir下所有以.npy结尾的文件
    raw_data_list = [
        log_dir + file_name for file_name in file_names if file_name.endswith('.npy')]
    positive_samples = []
    negative_samples = []
    neg_pos_ratio = 0
    train_neg_num = 0
    for k in range(len(raw_data_list)):
        raw_data_path = raw_data_list[k]
        processed_data = np.load(raw_data_path, allow_pickle=True)
        """
        if k>34:
            break
        """
        print(len(positive_samples))
        for step_data in processed_data:
            new_step_data = {}
            new_step_data['state'] = step_data['state']
            new_step_data['terrain'] = step_data['terrain']
            new_step_data['pre_terrain'] = step_data['pre_terrain']
            new_step_data['env_action'] = step_data['env_action']
            new_step_data['label'] = step_data['label']
            new_step_data['fall_dist'] = step_data['fall_dist']
            
            if (step_data['label']):
                positive_samples.append(new_step_data)
            
            """
            if not step_data['label']:
                negative_samples.append(step_data)
            else:
                positive_samples.append(step_data)
            """
        
        """
        if (k+1) % 5 == 0:
            np.save(f'/root/autodl-tmp/data/dataset/train_dataset_neg_{k}.npy', negative_samples, allow_pickle=True)
            print(f'save {k}')
            train_neg_num += len(negative_samples)
            print(f'train_neg_num:{train_neg_num}')
            negative_samples = []
        """
        print(f"finish process processed_data {k}/{len(raw_data_list)}")
    
    """
    for k in range(35,38):
        raw_data_path = raw_data_list[k]
        processed_data = np.load(raw_data_path, allow_pickle=True)
        val_samples_neg = []
        for step_data in processed_data:
            if not step_data['label']:
                # positive_samples.append(step_data)
                val_samples_neg.append(step_data)
                
        np.save(f'/root/autodl-tmp/data/dataset/val_dataset_neg.npy', val_samples_neg, allow_pickle=True)
        print(f"finish process processed_data {k + 1}/{len(raw_data_list)}")
    
    for k in range(38,len(raw_data_list)):
        raw_data_path = raw_data_list[k]
        processed_data = np.load(raw_data_path, allow_pickle=True)
        test_samples_neg = []
        for step_data in processed_data:
            if not step_data['label']:
                # positive_samples.append(step_data)
                test_samples_neg.append(step_data)
                
        np.save(f'/root/autodl-tmp/data/dataset/test_dataset_neg.npy', test_samples_neg, allow_pickle=True)
        
        print(f"finish process processed_data {k + 1}/{len(raw_data_list)}")
    """
    print(len(positive_samples))
    
    # neg_pos_ratio = int(len(negative_samples) / len(positive_samples))
    print(f'finish processing! number of positive samples is {len(positive_samples)},'
          f'number of negative samples is {len(negative_samples)}')

    # sampled_negative_samples = random.sample(negative_samples,2*len(positive_samples))
    # print(len(sampled_negative_samples))
    # np.save(f'../data/dataset/positive_samples.npy', positive_samples, allow_pickle=True)
    # np.save(f'../data/dataset/negative_samples.npy', sampled_negative_samples, allow_pickle=True)
    # np.save(f'../data/dataset/total_neg_samples.npy', negative_samples, allow_pickle=True)
    
    """
    num_train = int(p*len(positive_samples))
    num_test = int(0.5*(1-p)*len(positive_samples))
    print(num_train,num_test)
    """
    #train_samples_pos = positive_samples[:num_train]
    #train_samples_neg = negative_samples[:neg_pos_ratio*num_train]
    #val_samples_pos = positive_samples[num_train:num_train+num_test]
    #val_samples_neg = negative_samples[neg_pos_ratio *num_train:neg_pos_ratio*(num_train+num_test)]
    #test_samples_pos = positive_samples[num_train+num_test:]
    #test_samples_neg = negative_samples[neg_pos_ratio*(num_train+num_test):]
    np.save(f'/home/yjx/tta_new/data/dataset/train_dataset_pos_narrow_2.npy', positive_samples, allow_pickle=True)
    #np.save(f'/home/yjx/tta_new/data/dataset/train_dataset_neg.npy', train_samples_neg, allow_pickle=True)
    #np.save(f'/home/yjx/tta_new/data/dataset/val_dataset_neg_new.npy', val_samples_pos, allow_pickle=True)
    #np.save(f'/home/yjx/tta_new/data/dataset/val_dataset_neg.npy', val_samples_neg, allow_pickle=True)
    #np.save(f'/home/yjx/tta_new/data/dataset/test_dataset_neg_new.npy', test_samples_pos, allow_pickle=True)
    #np.save(f'/home/yjx/tta_new/data/dataset/test_dataset_neg.npy', test_samples_neg, allow_pickle=True)
    
    """
    print(f'Length of neg train samples is {len(train_neg_num)},'
          f'Length of neg val samples is {len(val_samples_neg)},'
          f'Length of neg test samples is {len(test_samples_neg)}')
    """

# 重新写一个可以处理正/负step样本的函数
def build_dataset_pn(log_dir,p,is_pos):
    # log_dir = 'tta/data/processed_data/'
    # p: proportion of train dataset 0.7
    file_names = os.listdir(log_dir)
    # 寻找log_dir下所有以.npy结尾的文件
    raw_data_list = [
        log_dir + file_name for file_name in file_names if file_name.endswith('.npy')]
    positive_samples = []
    negative_samples = []
    neg_pos_ratio = 0
    train_neg_num = 0
    for k in range(len(raw_data_list)):
        raw_data_path = raw_data_list[k]
        processed_data = np.load(raw_data_path, allow_pickle=True)
        if is_pos:
            print(len(positive_samples))
        else:
            print(len(negative_samples))
        for step_data in processed_data:
            new_step_data = {}
            new_step_data['state'] = step_data['state']
            new_step_data['terrain'] = step_data['terrain']
            new_step_data['pre_terrain'] = step_data['pre_terrain']
            new_step_data['env_action'] = step_data['env_action']
            new_step_data['label'] = step_data['label']
            new_step_data['fall_dist'] = step_data['fall_dist']
            
            if not step_data['label'] and (not is_pos):
                negative_samples.append(step_data)
            elif is_pos:
                positive_samples.append(step_data)
            else:
                pass
            
        print(f"finish process processed_data {k}/{len(raw_data_list)}")
    

    if is_pos:
        print(len(positive_samples))
        print(f'finish processing! number of positive samples is {len(positive_samples)}')
        np.save(f'/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data/positive_samples.npy', positive_samples, allow_pickle=True)
    else:
        print(len(negative_samples))
        print(f'finish processing! number of negative samples is {len(negative_samples)}')
        np.save(f'/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data/negative_samples.npy', negative_samples, allow_pickle=True)

    
# log_dir = '/home/yjx/tta_new/data/processed_data_pos/'
log_dir_pos = '/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/raw_data/pos/crash/'
log_dir_neg = '/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/raw_data/neg/'
build_dataset_pn(log_dir_pos,p=0.95,is_pos=1)
build_dataset_pn(log_dir_neg,p=0.95,is_pos=0)