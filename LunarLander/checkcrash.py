import numpy as np

# 替换成你的文件路径
file_path = '/home/teamcommon/tyy/MyLander/test_results/process_0/d2rl_new_2000.npy'

# 加载文件
data = np.load(file_path, allow_pickle=True)

#如果有某个数据>0，则print出来
for i in range(len(data)):
    if data[i] > 0:
        print(i, data[i])