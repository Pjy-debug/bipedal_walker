import numpy as np
# 加载.npy文件
data = np.load('/home/teamcommon/pjy/Bipedal_walker/criticality/data/processed_by_stage1/FP_samples.npy', allow_pickle=True)
# 查看数组的形状
print("数组形状:", data.shape)
# 查看数组的数据类型
print("数组数据类型:", data.dtype)
# 查看数组的维度
print("数组维度:", data.ndim)
# 查看
print("data[0]:", data[0])

