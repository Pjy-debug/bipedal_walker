from gymnasium.envs.registration import register
 
register(
    id="LunarLander/ordinary-v0",
    entry_point="gym_testenvs.LunarLander:LunarLander_ordinary",
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLander/train-v0",
    entry_point="gym_testenvs.LunarLander:LunarLander_train",
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLander/ordinary_nade-v0",
    entry_point="gym_testenvs.LunarLander:LunarLander_ordinary_nade",
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLander/ordinary_nade-v1",
    entry_point="gym_testenvs.LunarLander:LunarLander_nade",
    max_episode_steps=1000,
    reward_threshold=200,
)
register(
    id="LunarLander/ordinary_d2rl-v0",
    entry_point="gym_testenvs.LunarLander:LunarLander_ordinary_d2rl",
    max_episode_steps=1000,
    reward_threshold=200,
)
register(
    id="LunarLander/nde-v0",
    entry_point="gym_testenvs.LunarLander:LunarLander_nde",
    max_episode_steps=1000,
    reward_threshold=200,
)