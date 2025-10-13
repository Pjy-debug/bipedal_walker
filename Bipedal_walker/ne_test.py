import os
import time
import numpy as np
from tqdm import tqdm
from tta.niches.box2d.model import Model
from tta.niches.box2d.env import make_env, bipedhard_custom
from tta.niches.box2d.bipedal_walker_custom import BipedalWalkerCustom, Env_config

log_dir = "logs/poet_new_test/"
# log_dir = "logs/poet_final_test/"
best_model_final_test = "logs/poet_final_test/poet_final_test.48542109-e29d-4fbe-8405-6618c92e990a.best.json"
best_model_new_test = "logs/poet_new_test/poet_new_test.966593ea-181f-42ec-ad1d-2ca11487bea9.best.json"
file_names = os.listdir(log_dir)
best_models = [log_dir + file_name for file_name in file_names if file_name.endswith('.json')]
best_models = sorted(best_models, reverse=True)
model = Model(bipedhard_custom)

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

DUBUG = False
seed = 42
env = make_env("BipedalWalkerCustom", seed, DUBUG, env_config)

best_model = best_models[28]
model.load_model(best_model)

num = 100_000
max_iters = 200
cum_reward = []
trange = tqdm(range(num))
collected_data = []
# 运行trange次
for i in trange:
    observation = env.reset(seed=i)
    done = False
    k, r = 0, 0
    res = []
    while not done and k <= max_iters:
        k += 1
        action = model.get_action(observation)
        next_observation, reward, done, info = env.step(action)
        r += reward
        if DUBUG: env.render()
        # 记录
        res.append((observation,action,reward,next_observation,done))
        observation = next_observation
    cum_reward.append(r)
    collected_data.append(res)
    trange.set_description(f"r = {round(r, 2)}")
    if i % 1000 == 0:
        np.savetxt(f"data/cum_reward_{i}.csv", cum_reward, delimiter=",", fmt="%.4f")
        with open(f'data/raw_data/transitions_{i}.json', 'w', encoding='utf8') as f:
            json.dump(collected_data, f, ensure_ascii=False, indent=2)
        collected_data = []
env.close()
