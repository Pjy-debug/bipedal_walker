from data_utils import create_my_step_dataset, process_step_data

process_step_data('/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/raw_data/neg', 'neg')
create_my_step_dataset('/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data/processed_step_data_neg.npy', 'neg')
process_step_data('/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/raw_data/pos', 'pos')
create_my_step_dataset('/home/teamcommon/pjy/Bipedal_walker/criticality/data/collect_data_new/processed_data/processed_step_data_pos.npy', 'pos')
