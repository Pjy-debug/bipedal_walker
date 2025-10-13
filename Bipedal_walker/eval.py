import os
import time
import numpy as np
from tta.niches.box2d.model import Model
from tta.niches.box2d.env import make_env, bipedhard_custom
from tta.niches.box2d.bipedal_walker_custom import BipedalWalkerCustom, Env_config
from tta.niches.box2d.bipedal_walker_test import BipedalWalkerTest

log_dir = "logs/poet_new_test/"
# log_dir = "logs/poet_final_test/"
best_model_final_test = "logs/poet_final_test/poet_final_test.48542109-e29d-4fbe-8405-6618c92e990a.best.json"
best_model_new_test = "logs/poet_new_test/poet_new_test.966593ea-181f-42ec-ad1d-2ca11487bea9.best.json"
file_names = os.listdir(log_dir)
best_models = [
    log_dir + file_name for file_name in file_names if file_name.endswith('.json')]
best_models = sorted(best_models, reverse=True)
model = Model(bipedhard_custom)


env_config_0 = Env_config(
    name='default_env',
    ground_roughness=0,
    pit_gap=[],
    stump_width=[],
    stump_height=[],
    stump_float=[],
    stair_height=[],
    stair_width=[],
    stair_steps=[])

env_config_1 = Env_config(
    name="edge",
    ground_roughness=0.6,
    pit_gap=[0, 0.8],
    stump_width=[1, 2],
    stump_height=[0.01, 0.4],
    stump_float=[0, 1],
    stair_height=[],
    stair_width=[],
    stair_steps=[])

env_config_2 = Env_config(
    name="edge",
    ground_roughness=0,
    pit_gap=[2],
    stump_width=[],
    stump_height=[],
    stump_float=[],
    stair_height=[0, 2],
    stair_width=[1, 5],
    stair_steps=[1, 10])

# 应该是这个
ENV_CONFIG = env_config_1
RENDER_MODE = True
RGB_ARRAY = False
SAVE_REWARD = False
seed = 42
best_model = best_models[39]
model.load_model(best_model)
# env = make_env("BipedalWalkerCustom", seed, RENDER_MODE, ENV_CONFIG)
env = make_env("BipedalWalkerTest", seed, RENDER_MODE,
               ENV_CONFIG, model.get_action)
# time.sleep(3)

cum_reward = []
frames = []
for k in [40-1]:
    # for k in range(len(best_models)):
    observation = env.reset(seed=seed)
    ternimated = False
    i, r = 0, 0
    max_iters = 2000
    while not ternimated and i <= max_iters:
        i += 1
        # print(f"env.terrain_state for action = {env.terrain_state}")
        ENV_ACT = env.env_act
        if ENV_ACT:
            if env.terrain_state == 0:
                action = np.random.randint(5, 15)
            else:
                action = np.random.randint(5)
        else:
            action = None
        observation, reward, ternimated, truncated, info = env.step(action)

        if ternimated:
            print(info)
        if ENV_ACT or i % 10 == 0:
            print(i, round(reward, 2), round(r, 2), round(
                observation[2], 2), action, round(info["position"][0], 3))
        r += reward
        if RGB_ARRAY:
            frames.append(env.render(mode="rgb_array"))
        elif RENDER_MODE:
            env.render("human")
    cum_reward.append(r)
    if RGB_ARRAY:
        np.save(f"data/frames_{k}", frames)
env.close()

if SAVE_REWARD:
    np.savetxt("data/cum_reward.csv", cum_reward, delimiter=",", fmt="%.4f")
