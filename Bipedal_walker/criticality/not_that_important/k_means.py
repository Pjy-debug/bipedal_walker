from sklearn.cluster import MiniBatchKMeans
import numpy as np


#dataset_paths_neg = [4,9,15,19,24,29]
dataset_path_neg = '/root/autodl-tmp/data/dataset/train_dataset_neg_4.npy'
batch_size = 100000
n_clusters = 8000
print('loading data...')
train_data_neg = np.load(dataset_path_neg, allow_pickle=True)
train_data_neg = list(train_data_neg)
print('Successfully load data',len(train_data_neg))
inputs = []
labels = []
for k, step_data in enumerate(train_data_neg):
    labels.append((step_data['label'],step_data['fall_dist']))
    terrain = []
    for pre_terrain in step_data['pre_terrain']:
        for item in pre_terrain:
            terrain += list(item)
    for item in step_data['terrain']:
        terrain += list(item)
    # 去掉pos
    input = list(step_data['state']) + terrain + [step_data['env_action']]
    #print(len(input))
    inputs.append(input)
print('finish processing!')
inputs = np.array(inputs)
max_iter = len(labels)//batch_size
print(f'max_iter:{max_iter}')
kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                         random_state=0,
                         batch_size=batch_size)
k = 0
total_centers = []
while k < max_iter-1:
    print(f'processing {k*batch_size} to {(k+1)*batch_size}')
    X = inputs[k*batch_size:(k+1)*batch_size,:]
    kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                             random_state=0,
                             batch_size=batch_size,
                             max_iter=50).fit(X)
    #kmeans = kmeans.partial_fit(inputs[k*batch_size:(k+1)*batch_size,:])
    centers = kmeans.cluster_centers_
    print(len(centers))
    total_centers.append(centers)
    k += 1

np.save(f'/root/brx/tta_new/tta/criticality/new_log/result/data_neg_6.npy', total_centers, allow_pickle=True)
print(f'centers:{list(centers)[:10]}')

"""
kmeans = MiniBatchKMeans(n_clusters=100000,
                         random_state=0,
                         batch_size=2048,
                         max_iter=100).fit(inputs)
centers = kmeans.cluster_centers_
np.save(f'/root/brx/tta_new/tta/criticality/new_log/result/data_neg_1.npy', centers, allow_pickle=True)
print(f'centers:{list(centers)[:100]}')
"""