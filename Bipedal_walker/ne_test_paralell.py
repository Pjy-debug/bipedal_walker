'''*************************************************************************
【文件名】                 （必需）
【功能模块和目的】         训练BipedalWalker模型，作为被测模型
【开发者及日期】           （必需）
【更改记录】               （若修改过则必需注明）
*************************************************************************'''
import os
import numpy as np
from utils.seeding import generate_seed
import datetime
import multiprocessing as mp
from tqdm import tqdm
from tta.niches.box2d.model import Model
from tta.niches.box2d.env import make_env, bipedhard_custom
from tta.niches.box2d.bipedal_walker_custom import BipedalWalkerCustom, Env_config
import warnings
warnings.filterwarnings('ignore')


def ne_test(num, env_config, max_iters=2000, debug=False):
    cum_reward = []
    log_position = []
    trange = tqdm(range(int(num)))
    env = make_env("BipedalWalkerCustom", 34, render_mode=False, env_config=env_config)
    for i in trange:
        observation = env.reset(seed=generate_seed())
        done = False
        k, r = 0, 0
        while not done and k <= max_iters:
            k += 1
            action = model.get_action(observation)
            observation, reward, done, info = env.step(action)
            r += reward
            if debug: env.render()
        log_position.append(info["position"][0])
        cum_reward.append(r)
        if debug:
           trange.set_description(f"r = {round(r, 2)}")
        env.close()
    return cum_reward, log_position


if __name__ == '__main__':

    start_t = datetime.datetime.now()
    num_cores = int(mp.cpu_count())
    print("total available threads = " + str(num_cores))

    log_dir = "logs/poet_new_test/" # "logs/poet_final_test/"
    file_names = os.listdir(log_dir)
    best_models = [log_dir + file_name for file_name in file_names if file_name.endswith('.json')]
    best_models = sorted(best_models, reverse=True)
    best_model = best_models[-1]
    model = Model(bipedhard_custom)
    model.load_model(best_model)

    env_config = Env_config(
            name="edge",
            ground_roughness=0.6,
            pit_gap=[0, 0.8],
            stump_width=[1, 2],
            stump_height=[0.01, 0.4],
            stump_float=[0, 1],
            stair_height=[],
            stair_width=[],
            stair_steps=[])

    num_cores_use = 50
    num_test_per_core = 1000_0000 // num_cores_use
    pool = mp.Pool(num_cores_use)
    results = [pool.apply_async(ne_test, args=(num, env_config,)) for num in np.ones(num_cores_use) * num_test_per_core]
    episode_rewards = np.array([p.get()[0] for p in results])
    positions = np.array([p.get()[1] for p in results])
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    np.save('data/rewards_1e7_0', episode_rewards.flatten())
    np.save('data/positions_1e7_0', positions.flatten())
    print("total time spent = " + "{:.2f}".format(elapsed_sec / 60) + " min")
