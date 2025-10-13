import os
import time
import numpy as np
from tta.niches.box2d.model import Model
from tta.niches.box2d.env import make_env, bipedhard_custom
from tta.niches.box2d.bipedal_walker_custom import BipedalWalkerCustom, Env_config
from tta.niches.box2d.cppn import CppnEnvParams

log_dir = "logs/poet_new_test/"
# log_dir = "logs/poet_final_test/"
best_model_final_test = "logs/poet_final_test/poet_final_test.48542109-e29d-4fbe-8405-6618c92e990a.best.json"
best_model_new_test = "logs/poet_new_test/poet_new_test.966593ea-181f-42ec-ad1d-2ca11487bea9.best.json"
file_names = os.listdir(log_dir)
best_models = [log_dir + file_name for file_name in file_names if file_name.endswith('.json')]
best_models = sorted(best_models, reverse=True)
model = Model(bipedhard_custom)

env = Env_config(
        name='default_env',
        ground_roughness=0,
        pit_gap=[],
        stump_width=[],
        stump_height=[],
        stump_float=[],
        stair_height=[],
        stair_width=[],
        stair_steps=[])

seed = 42222
env = make_env("BipedalWalkerCustom", seed, True, env)
# params = CppnEnvParams(genome_path="/tmp/genome_1678451422.825029.pickle")
# params = CppnEnvParams(genome_path="/tmp/genome_1678451467.5480528.pickle")
# params = CppnEnvParams(genome_path="/tmp/genome_1678456581.0893025.pickle")
# params = CppnEnvParams(genome_path="/tmp/genome_1678470346.9981463.pickle")
# params = CppnEnvParams(genome_path="/tmp/genome_1678490177.5360248.pickle")
# params = CppnEnvParams(genome_path="/tmp/genome_1678511100.2262144.pickle")
# params = CppnEnvParams(genome_path="/tmp/genome_1678532841.4572017.pickle")
# params = CppnEnvParams(genome_path="/tmp/genome_1678684122.6467571.pickle")
# params = CppnEnvParams(genome_path="/tmp/genome_1678655385.4286368.pickle")
# params = CppnEnvParams(genome_path="/tmp/genome_1678628399.7841783.pickle") # ok
# params = CppnEnvParams(genome_path="/tmp/genome_1678628411.9550507.pickle")
# params = CppnEnvParams(genome_path="/tmp/genome_1678628387.7907777.pickle") # ok
# params = CppnEnvParams(genome_path="/tmp/genome_1678628375.5103068.pickle") # ok
# params = CppnEnvParams(genome_path="/tmp/genome_1678628349.7523198.pickle") # ok
# params = CppnEnvParams(genome_path="/tmp/genome_1678628338.4884624.pickle") # ok
# params = CppnEnvParams(genome_path="/tmp/genome_1678603053.0863962.pickle") # ok
# params = CppnEnvParams(genome_path="/tmp/genome_1678603040.9751065.pickle") # ok
# params = CppnEnvParams(genome_path="/tmp/genome_1678779646.5285704.pickle") # ok
# params = CppnEnvParams(genome_path="/tmp/genome_1678745768.8832457.pickle") # ok
# params = CppnEnvParams(genome_path="/tmp/genome_1678713840.6359096.pickle") # ok
# params = CppnEnvParams(genome_path="/tmp/genome_1678745755.7463415.pickle") # ok
params = CppnEnvParams(genome_path="/tmp/genome_1678745670.7718053.pickle") # ok

# params.get_mutated_params()
env.augment(params)
best_model = best_models[-1]
model.load_model(best_model)
Obs = []
observation = env.reset(seed=seed)
done = False
i, r = 0, 0
RGB_ARRAY = False
max_iters = 3000
frames = {}
while not done and i <= max_iters:
    i += 1
    action = model.get_action(observation)
    observation, reward, done, info = env.step(action)
    r += reward
    if i % 10 == 0:
        print(i, round(reward, 2), round(r, 2), action)
    if RGB_ARRAY:
        frames[i] = env.render('rgb_array')
    else:
        env.render()
env.close()

if RGB_ARRAY: np.save("data/frames", frames)

