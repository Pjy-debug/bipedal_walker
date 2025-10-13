import numpy as np

# 替换为你的文件路径
file_path = '/home/teamcommon/tyy/MyLander/Env_agent/stage1/test_FP_final5.npy'
data = np.load(file_path, allow_pickle=True)  # 关键：必须加 allow_pickle=True
data = list(data)
# 查看第一条数据
print(data[0])  # 会打印完整的字典内容