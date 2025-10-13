import numpy as np
#dataset_paths_neg = [4,9,15,19,24,29]
dataset_paths_neg = [f'/root/brx/tta_new/tta/criticality/new_log/result/data_neg_{k}.npy' for k in range(6,7)]
samples_data_neg = []
for dataset_path_neg in dataset_paths_neg:
    print(dataset_path_neg)
    train_data_neg = list(np.load(dataset_path_neg, allow_pickle=True))
    #sample_data_neg = random.sample(train_data_neg,int(len(samples_data_pos)/len(dataset_paths_neg)))
    print(len(list(train_data_neg)))
    print(len(list(train_data_neg)[0]))
    train_data_neg = list(train_data_neg)
    for item in train_data_neg:
        #print(len(list(item)))
        #print(list(item))
        samples_data_neg += list(item)
print(len(samples_data_neg))
print(samples_data_neg[0])
np.save(f'/root/brx/tta_new/tta/criticality/new_log/result/train_dataset_neg_6.npy', samples_data_neg, allow_pickle=True)