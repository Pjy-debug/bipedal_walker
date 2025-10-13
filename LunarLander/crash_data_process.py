import pickle
import os
import numpy as np
import numpy.linalg as LA

data_dir = "/mnt1/brx/rd/scenario_data"
files = os.listdir(data_dir)
print(len(files))
combined_traj_list = []
label_list = []
mask_list = []
fixed_len =  15 #20 #10
pos = 0
neg = 0
for i, fn in enumerate(files):
    #print(i,len(files))
    fn = os.path.join(data_dir, fn)
    with open(fn, "rb") as f:
        data = pickle.load(f)
        #print(data.keys())
    
        location = data['location']    #[num_cars, time_step, 2]
        speed_heading = data['speed_heading'] #[num_cars, time_step, 1]
        mask = data['valid_mask']
        label_mask = data['crash_mask']
        if np.sum(label_mask):
          pos += 1
        else:
          neg += 1
        
        #print(np.sum(mask, axis=-1))
        sel_idx = (np.sum(mask, axis=-1)>=15) == 1
        shuffled_location = location[sel_idx]
        shuffled_speed_heading = speed_heading[sel_idx]
        shuffled_label_mask = label_mask[sel_idx]
       
        #label = data['label']
        
        location_arr = shuffled_location[:, -fixed_len:,:]
        heading_arr = shuffled_speed_heading[:, -fixed_len:][
            ..., np.newaxis
        ]
        combined_arr = np.concatenate([location_arr, heading_arr], axis=-1) #[num_cars, time_step, 3]
        
        velocity_2d = np.diff(combined_arr[..., :2], axis=2)
        velocity = LA.norm(velocity_2d, axis=-1)
        
        combined_arr = combined_arr
        label_mask = shuffled_label_mask
        
        if np.sum(label_mask):
          continue
        
        num_cars = combined_arr.shape[0]
        
        if (combined_arr.shape[0] > 32) or (combined_arr.shape[0] <=0 ) :
            continue
        elif combined_arr.shape[0] == 32:
            combined_traj_list.append(combined_arr)
            mask = np.array([False] * 32)
            label = label_mask
        else:
            #print(combined_arr.shape)
            pad = np.zeros((32-combined_arr.shape[0], combined_arr.shape[1], combined_arr.shape[-1]))
            label_pad = np.array([False]*(32-num_cars))
            label = np.concatenate([label_mask, label_pad], axis=0)
            combined_traj_list.append(np.concatenate([combined_arr, pad], axis=0))
            mask = np.array([False]*num_cars + [True]*(32-num_cars))
        
        mask_list.append(mask)
        label_list.append(label)
        
        
        # input: [bs, 32, 5x3] ---- output: [bs, 32, 1]

# filter huge jump trajectories
print('get list,',len(combined_traj_list))
print(pos,neg)

combined_traj_arr = np.stack(combined_traj_list, axis=0) #[num_samples, num_cars, time_step, 3]
print(combined_traj_arr.shape)

src_mask = np.stack(mask_list, axis=0)
print(src_mask.shape)

labels = np.stack(label_list, axis=0)
print(labels.shape)

num_samples = combined_traj_arr.shape[0]

np.save('/mnt1/brx/rd/safe_data.npy', combined_traj_arr)
np.save('/mnt1/brx/rd/safe_data_mask.npy', src_mask)
np.save('/mnt1/brx/rd/safe_label.npy', labels)