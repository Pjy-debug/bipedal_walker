from criticality_mc import bipedal_walker_criticality,calculate_val, calu_true_criticality
import numpy as np

def collect_true_criticality(log_dir,p):
    # log_dir = 'tta/data/processed_data/'
    # p: proportion of train dataset 0.7
    file_names = os.listdir(log_dir)
    raw_data_list = [
        log_dir + file_name for file_name in file_names if file_name.endswith('.npy')]
        
    for k in range(len(raw_data_list)):
        raw_data_path = raw_data_list[k]
        processed_data = np.load(raw_data_path, allow_pickle=True)
        
        for item in processed_data:
            failure_rate = calu_true_criticality(item)
            print(failure_rate)
            # trajectory['episode'].append(list(failure_rate))
            item['failure_rate'] = np.array(failure_rate)

            if item['terrain_state'] == 0:
                p_list = [0.1] * 10
                p_list = np.array(p_list)
                
            else:
                p_list = [0.2] * 5

collect_true_criticality('data/mc_data/')