# import gymnasium as gym
# env = gym.make("LunarLander-v2", render_mode="human")
# observation, info = env.reset(seed=42)
# for _ in range(1000):
#    action = env.action_space.sample()  # this is where you would insert your policy
#    observation, reward, terminated, truncated, info = env.step(action)

#    if terminated or truncated:
#       observation, info = env.reset()
# env.close()

from tta.niches.box2d.bipedal_walker_custom import BipedalWalkerCustom, Env_config

name_list = ["flat"]
name = name_list[0]
env_config = Env_config(
    name="edge",
    ground_roughness=10,
    pit_gap=[3],
    stump_width=[0, 0.8],
    stump_height=[0, 0.4],
    stump_float=[],
    stair_height=[0, 0.4],
    stair_width=[],
    stair_steps=[1])

env = BipedalWalkerCustom(env_config)
env.render("human")
observation = env.reset(seed=42)
for i in range(10000):
    action = env.action_space.sample()  # this is where you would insert your policy
    print(i, action)
    observation, reward, done, info = env.step(action)
    #print(info['agent_i'],info['pred_i'])
    env.render("human")
    if done:
        observation = env.reset()
env.close()
