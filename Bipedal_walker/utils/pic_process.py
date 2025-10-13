'''*************************************************************************
【文件名】                 pic_process.py
【功能模块和目的】         pic_process.py用于将渲染的npy文件转换为gif文件
【开发者及日期】           （必需）
【更改记录】               （若修改过则必需注明）
*************************************************************************'''
import os

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


def frames_to_gif(frames, save_dir, fps=3000):
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
    anim.save(save_dir, writer='imagemagick', fps=fps)

experiment_name = 'test_2025-3-19'
frames_path = os.listdir(f'render/{experiment_name}')
# remove the non npy files
frames_path = [path for path in frames_path if path.endswith('.npy')]
# rank by index
frames_path.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
print(frames_path)
for k, path in enumerate(frames_path):
    frame_path = 'render/'+path
    print("frame_path",frame_path)
    frames = np.load(frame_path,allow_pickle=True)
    save_dir = f'render/gif_{k}.gif'
    print(save_dir)
    frames_to_gif(frames,save_dir)