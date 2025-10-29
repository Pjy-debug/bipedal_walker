import numpy as np

def conreact_neg(source_path, target_path):
    source_data = np.load(source_path, allow_pickle=True)
    total_len=len(source_data)
    target_data=source_data[0:total_len//10]
    np.save(target_path, target_data, allow_pickle=True)

source_path = '/home/teamcommon/pjy/Bipedal_walker/criticality/data/processed_by_stage1/FP_samples.npy'
target_path = '/home/teamcommon/pjy/Bipedal_walker/criticality/data/processed_by_stage1/FP_samples_contracted.npy'
conreact_neg(source_path, target_path)