import matplotlib.pyplot as plt
from matplotlib import animation


def frames_to_gif(frames, save_dir, fps=30):
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
    anim.save(save_dir, writer='imagemagick', fps=fps)
