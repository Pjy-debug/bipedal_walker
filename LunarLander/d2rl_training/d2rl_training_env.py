from gym import spaces, core
import os, glob
import random
import json
import numpy as np
import logging
import yaml


class D2RLTrainingEnv(core.Env):
    def __init__(self, yaml_conf):
        #data_folders = [yaml_conf["root_folder"] + folder for folder in yaml_conf["data_folders"]]
        data_folders = ['/mnt1/brx/Rocketdata/d2rl_data']
        data_folder_weights = yaml_conf["data_folder_weights"]
        self.yaml_conf = yaml_conf
        self.action_space = spaces.Box(low=0.001, high=0.999, shape=(1,))
        self.observation_space = spaces.Box(low=-5, high=5, shape=(8,))

        self.constant, self.weight_reward, self.exposure, self.positive_weight_reward = 0, 0, 0, 0  # some customized metric logging
        self.total_episode, self.total_steps = 0, 0
        if isinstance(data_folders, list):
            data_folder = random.choices(data_folders, weights=data_folder_weights)[0]
        else:
            data_folder = data_folders
        self.crash_data_path_list, self.crash_data_weight_list, self.crash_target_weight_list = self.get_path_list(data_folder)
        self.all_data_path_list = self.crash_data_path_list
        self.episode_data_path = ""
        self.episode_data = None

        self.unwrapped.trials = 100
        self.unwrapped.reward_threshold = 1.5

    def get_path_list(self, data_folder):
        crash_target_weight_list = None
        
        #print(data_folder + "/crash_weight_dict.json")
        if os.path.exists(data_folder + "/crash_weight_dict.json"):
            with open(data_folder + "/crash_weight_dict.json") as data_file:
                crash_weight_dict = json.load(data_file)
                crash_data_path_list = list(crash_weight_dict.keys())
                crash_data_weight_list = [crash_weight_dict[path] for path in crash_data_path_list]
        else:
            raise ValueError("No weight information!")
        #print('get_path_list--crash_data_path_list:',crash_data_path_list)
        #print(crash_data_weight_list)
        return crash_data_path_list, crash_data_weight_list, crash_target_weight_list

    def reset(self, episode_data_path=None):
        self.constant, self.weight_reward, self.exposure, self.positive_weight_reward = 0, 0, 0, 0
        self.total_episode = 0
        self.total_steps = 0
        self.episode_data_path = ""
        self.episode_data = None
        return self._reset(episode_data_path)

    def filter_episode_data(self, episode_data):
        invalid_timestep_list = []
        #print(episode_data.keys())
        for timestep in episode_data["weight_step_info "]:
            if episode_data["weight_step_info "][timestep] < 1.001 and episode_data["weight_step_info "][timestep] > 0.99:     # 存疑
                invalid_timestep_list.append(timestep)
                logging.debug(f"popping out {episode_data['weight_step_info ']}")
        for invalid_time_step in invalid_timestep_list:
            episode_data["weight_step_info "].pop(invalid_time_step, None)
            episode_data["drl_epsilon_step_info "].pop(invalid_time_step, None)
            episode_data["real_epsilon_step_info"].pop(invalid_time_step, None)
            episode_data["criticality_step_info"].pop(invalid_time_step, None)
            episode_data["ndd_step_info"].pop(invalid_time_step, None)
            episode_data["drl_obs_step_info"].pop(invalid_time_step, None)
        # logging.debug(str(episode_data))
        return episode_data

    def sample_data_this_episode(self):
        print('sampling data path')
        if self.crash_data_weight_list:
            #print('sample_data--data_path_list:',self.crash_data_path_list)
            #print('sample_data--weight_list:',self.crash_data_weight_list)
            episode_data_path = random.choices(self.crash_data_path_list, weights=self.crash_data_weight_list)[0]
        else:
            raise ValueError("No weight information!")
        #print('episode_data_path:',episode_data_path,type(episode_data_path))
        return episode_data_path

    def _reset(self, episode_data_path=None):
        print('resetting')
        self.total_episode += 1
        if not episode_data_path:
            self.episode_data_path = self.sample_data_this_episode()
        else:
            self.episode_data_path = episode_data_path

        #print('reset--episode data path:',self.episode_data_path)

        self.episode_data_path =  self.episode_data_path
        #print(self.episode_data_path)
        #self.crash_data_path_list = self.crash_data_weight_list
        with open(self.episode_data_path) as data_file:
            self.episode_data = self.filter_episode_data(json.load(data_file))
        if self.episode_data is not None:
            all_obs = self.episode_data["drl_obs_step_info"]
            time_step_list = list(all_obs.keys())
            if len(time_step_list):
                init_obs = np.float32(all_obs[time_step_list[0]])
                return init_obs
            else:
                return self._reset()
        else:
            return self._reset()

    def step(self, action):
        print('stepping!')
        #print(self.total_episode, self.total_steps, self.episode_data_path)
        action = action.item()  # ok
        obs = self._get_observation()   # ok
        done = self._get_done()  # ok
        time_step_list = list(self.episode_data["drl_obs_step_info"].keys())    # ok
        criticality_this_step = self.episode_data["criticality_step_info"][time_step_list[self.total_steps]]
        self.episode_data["drl_epsilon_step_info "][time_step_list[self.total_steps]] = action
        reward = self._get_reward() # ok
        info = self._get_info()
        self.total_steps += 1
        return obs, reward, done, info

    def _get_info(self):
        return {}

    def close(self):
        return

    def _get_observation(self):
        all_obs = self.episode_data["drl_obs_step_info"]
        time_step_list = list(all_obs.keys())
        try:
            obs = np.float32(all_obs[time_step_list[self.total_steps]])
        except:
            print(self.total_steps, time_step_list)
            obs = np.float32(all_obs[time_step_list[-1]])
        return obs

    def get_multiple_adv_action_num(self, weight_info):
        adv_action_num = 0
        for timestep in weight_info:
            if weight_info[timestep] < 0.99:        # 存疑
                adv_action_num += 1
        return adv_action_num

    def _get_reward(self):  # ! Aim to remove the magnitude of the environment
        stop = self._get_done()
        if not stop:
            return 0
        else:
            drl_epsilon_weight = self._get_drl_epsilon_weight(self.episode_data["weight_step_info "],
                                                              self.episode_data["drl_epsilon_step_info "],
                                                              self.episode_data["ndd_step_info"],
                                                              self.episode_data["criticality_step_info"])
            print(self.episode_data["drl_epsilon_step_info "])
            adv_action_num = self.get_multiple_adv_action_num(self.episode_data["weight_step_info "])
            if adv_action_num > 1:
                return 0  # if multiple adversarial action is detected, this episode will be of no use
            clip_reward_threshold = self.yaml_conf["clip_reward_threshold"]
            q_amplifier_reward = clip_reward_threshold - drl_epsilon_weight * 500 * clip_reward_threshold  # drl epsilon weight reward
            if q_amplifier_reward < -clip_reward_threshold:
                q_amplifier_reward = -clip_reward_threshold
            print("final_reward:", q_amplifier_reward)

            return q_amplifier_reward

    def _get_drl_epsilon_weight(self, weight_info, epsilon_info, ndd_info, criticality_info=None):
        total_q_amplifier = 1
        for timestep in epsilon_info:
            if timestep in weight_info:
                if weight_info[timestep] > 1:
                    total_q_amplifier = total_q_amplifier * (1 / epsilon_info[timestep])
                elif weight_info[timestep] < 0.999:
                    if timestep not in ndd_info:
                        ndd_tmp = criticality_info[timestep]
                    else:
                        ndd_tmp = ndd_info[timestep]
                    total_q_amplifier = total_q_amplifier * (1 / (1 - epsilon_info[timestep])) * ndd_tmp
        return total_q_amplifier

    def _get_done(self):
        stop = False
        if self.total_steps == len(self.episode_data["drl_obs_step_info"].keys()) - 1:
            stop = True
        return stop


if __name__ == "__main__":
    f = open('d2rl_train.yaml', 'r', encoding='utf-8')
    yaml_conf = yaml.safe_load(f)

    env = D2RLTrainingEnv(yaml_conf)

    for i in range(100):
        print(f'episode {i},-------------------')
        obs = env.reset()
        while True:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                break