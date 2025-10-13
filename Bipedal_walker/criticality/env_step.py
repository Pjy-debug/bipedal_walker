import numpy as np

TERRAIN_STEP = 14/30
GRASS = 0
STUMP = 1
PIT = 2
STATES = 4
#ground_roughness=0.6

SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 20    # in steps
FRICTION = 2.5

'''*************************************************************************
[function name]             env_step
[function details]          env step with an action
[inputs]                    int | my_terrain_state:
                            int | my_action
                            int | my_current_i
                            int | my_pred_i
                            int | my_terrain_counter
                            int | terrain_x
                            float | my_terrain_velocity
                            float | my_current_y
                            dict | env_params
                            Env_config | config
                            np.random | np_random
                            bool | hardcore=True,
                            bool | sparse_IS=True
                            bool | debug=False
[outputs]                   int | p_orin: 1
                            list | cur_step_poly: list of poly (in tuple, x & y pos)
                            int | terrain_num: relative to terrain_state and terrain_counter
[developer&date]           （必需）
[change log]               （若修改过则必需注明）
*************************************************************************'''
def env_step(my_terrain_state, my_action, my_current_i, my_pred_i, my_terrain_counter, terrain_x,
             my_terrain_velocity, my_current_y, env_params,config,np_random,hardcore=True, sparse_IS=True, debug=False):
    #print('-------------my_env_step!--------------','terrain_state=',terrain_state)
    #print('current_i=',current_i,'pred_i=',pred_i)
    """
    action : env_action
    terrain: env.terrain
    current_i: env.current_i
    pred_i: env.pred_i
    terrain_counter: env.terrain_counter
    terrain_x: env.terrain_x
    GRASS: self.GRASS
    STATES: self._STATES_
    terrain_velocity:self.terrain_velocity
    current_y: self.current_y
    env_params: self.env_params
    config: self.config
    """
    terrain_state = my_terrain_state
    current_i = my_current_i
    pred_i = my_pred_i
    terrain_counter = my_terrain_counter
    terrain_velocity = my_terrain_velocity
    current_y = my_current_y
    init_i = my_current_i
    cur_step_poly = []
    p_orin = 1
    my_terrain_x = []
    input_action = my_action
    action = my_action
    first_terrain = True

    # if debug: print(f"init_i = {init_i}, pred_i = {pred_i}, pred_bound = {pred_bound}")
    # print('env take actions!')
    terrain_num = 0
    while current_i < pred_i or terrain_counter > 0:
    #while terrain_counter > 0:
        #print('current_i=',current_i,'pred_i=',pred_i)

        # 本次更新的开始x值
        x = current_i * TERRAIN_STEP
        my_terrain_x.append(x)
        current_i += 1
        
        
        if debug: print(f"i = {current_i}, x = {round(x, 2)}")
        #print('input_action:',action,'terrain_counter:',terrain_counter,'first_terrain:',first_terrain)
        
        if terrain_counter == 0:
            #print(f"self.terrain_state = {self.terrain_state}")
            # 草地
            if terrain_state == GRASS and hardcore:
                #print(f"action = {action},terrain_state={terrain_state}")
                if action <= 4:
                    if input_action < 10:
                        action += 5  
                    elif input_action >= 10:
                        action += 10
                assert action > 4 and action < 15
                if sparse_IS:
                    terrain_state = 1 if action > 9 else 2
                    terrain_num += 1                    
                    #print('terrain counter == 0 and terrain_state == grass, p=',p_orin,'action=',action,'first_terrain=',first_terrain)
                else:
                    # 下一个生成地形类别
                    terrain_state = np_random.integers(1, self._STATES_)
                if debug: print(f"self.terrain_state = {self.terrain_state}")
            else:
                # 坑、凸起后一定是草地
                #print(f"action = {action},terrain_state={terrain_state}")
                if action > 4 and action < 10:
                    action -= 5  # [0-4]
                elif action > 9:
                    action -= 10
                terrain_state = GRASS
                if sparse_IS:
                    # [5,10]
                    terrain_counter = action + 5
                    #p_orin *= 2 / TERRAIN_GRASS   # 10
                    #p = 1
                    terrain_num += 1
                    #print('terrain counter == 0 and terrain_state != grass, p=',p_orin,'first_terrain=',first_terrain)
                else:
                    terrain_counter = np_random.integers(TERRAIN_GRASS / 2, TERRAIN_GRASS)
                if debug: print(f"self.terrain_counter = {terrain_counter}")
            terrain_oneshot = True


        if terrain_state == GRASS and not terrain_oneshot:
            terrain_velocity = 0.8 * terrain_velocity + \
                 0.01 * np.sign(TERRAIN_HEIGHT - current_y)
            if env_params is not None and env_params.altitude_fn is not None:
                print('env_params is not none!')
                current_y += terrain_velocity
                if i > TERRAIN_STARTPAD:
                    mid = TERRAIN_LENGTH * TERRAIN_STEP / 2.
                    x_ = (x - mid) * np.pi / mid
                    current_y = TERRAIN_HEIGHT + env_params.altitude_fn((x_, ))[0]
                    if i == TERRAIN_STARTPAD+1:
                        y_norm = env_params.altitude_fn((x_, ))[0]
                    current_y -= y_norm
                if debug: print(f"cppn used, i = {self.current_i}, y = {self.current_y}")
            else:
                if sparse_IS:
                    # action来控制草地摩擦程度
                    # delta_v: [-1,1]
                    if action >=0 and action < 5:
                        delta_v = (action - 2 ) / 2
                    elif action >=5 and action < 15:
                        delta_v = (action - 9.5) / 4.5
                    terrain_velocity += delta_v  / SCALE
                    #print('GRASS=',GRASS,'terrain counter > 0 and terrain_state == grass, p=',p,'action=',action,'delta_v=',delta_v,'first_terrain=',first_terrain)
                else:
                    delta_v = np_random.uniform(-1, 1) / SCALE
                    terrain_velocity += delta_v

                current_y += config.ground_roughness * terrain_velocity
                if debug: print(f"y = {round(current_y, 2)}")

        elif terrain_state == PIT and terrain_oneshot:
            if sparse_IS:
                # 用action来控制坑的宽度 [0,0.8]
                # action: [0,4]
                pit_gap = 1.0 + (action - 5) * 0.2
                # 这是一个固定值，不用再乘p，完全由action控制
                p = 1 / (config.pit_gap[1] - config.pit_gap[0])
                #print('pit=',PIT,'terrain counter == 0 and terrain_state == git, p=', p,'pit_gap=',pit_gap,'action=',action)
            else:
                pit_gap = 1.0 + np_random.uniform(*config.pit_gap)
            if debug: print(f"pit_gap = {pit_gap}")

            terrain_counter = np.ceil(pit_gap)
            pit_diff = terrain_counter - pit_gap

            poly = [
                (x,                current_y),
                (x + TERRAIN_STEP, current_y),
                (x + TERRAIN_STEP, current_y - 4 * TERRAIN_STEP),
                (x,                current_y - 4 * TERRAIN_STEP),
            ]
            terrain_counter += 2
            cur_step_poly = [(po[0] + TERRAIN_STEP * pit_gap, po[1]) for po in poly]
            original_y = current_y

        elif terrain_state == PIT and not terrain_oneshot:
            current_y = original_y
            if terrain_counter > 1:
                current_y -= 4 * TERRAIN_STEP
            if terrain_counter == 1:
                my_terrain_x[-1] = my_terrain_x[-1] - pit_diff * TERRAIN_STEP
                pit_diff = 0

        elif terrain_state == STUMP and terrain_oneshot:
            if sparse_IS:
                # 用action来控制桩的高度
                # 1
                #stump_width = np_random.integers(*config.stump_width)
                stump_width = 1
                # action: [5,9]
                # [0,0.4]
                stump_height = (9 - action) * 0.1
                if stump_height < 0:
                    stump_height = -stump_height
                # 0
                #stump_float = np_random.integers(*config.stump_float)
                stump_float = 0
                #print('stump=',STUMP,'terrain counter == 0 and terrain_state == stump', 'p=',p,'width,height,float=',stump_width,stump_height,stump_float,'action=',action)
            else:
                stump_width = np_random.integers(*config.stump_width)
                stump_height = np_random.uniform(*config.stump_height)
                stump_float = np_random.integers(*config.stump_float)
            if debug:
                print(f"stump_width = {stump_width}, stump_height = {stump_height}, stump_float = {stump_float}")
            terrain_counter = stump_width
            countery = stump_height
            poly = [
                (x,                              current_y + stump_float * TERRAIN_STEP),
                (x + stump_width * TERRAIN_STEP, current_y + stump_float * TERRAIN_STEP),
                (x + stump_width * TERRAIN_STEP, current_y + countery *TERRAIN_STEP + stump_float * TERRAIN_STEP),
                (x,                              current_y + countery * TERRAIN_STEP + stump_float * TERRAIN_STEP),
            ]
            cur_step_poly = poly

        terrain_oneshot = False
        terrain_counter -= 1
        first_terrain = False

    # current_i = pred_i
    # print('start_i:',init_i,'end_i:',self.current_i,'total_step:',self.current_i-init_i)
    if not cur_step_poly:
        cur_step_poly = [(init_i*TERRAIN_STEP, current_y),(current_i * TERRAIN_STEP, current_y),
                         (init_i*TERRAIN_STEP, current_y),(current_i * TERRAIN_STEP, current_y)]
    #print('cur_step_poly',cur_step_poly)
    #print('p_orin',p_orin,'terrain_num',terrain_num)

    return p_orin, cur_step_poly, terrain_num

"""
def env_step(terrain_state, action, terrain, current_i, pred_i, terrain_counter, terrain_x,
             terrain_velocity, current_y, env_params,config,np_random,hardcore=True, sparse_IS=True, debug=False):
    
    init_i = current_i
    my_current_i = current_i
    my_pred_i = pred_i
    my_terrain_x = []
    cur_step_poly = []
    input_action = action
    my_terrain_counter = terrain_counter
    terrain_num = 0
    
    while current_i < pred_i or terrain_counter > 0:
        x = current_i * 14 / 30.0
        my_terrain_x.append(x)
        current_i += 1
        
        if terrain_counter == 0:
            # 当前生成草地
            if terrain_state == GRASS and hardcore:
                # print(f"action = {action}")
                # new
                if action <= 4:
                    if input_action < 10:
                        action += 5  
                    elif input_action >= 10:
                        action += 10
                assert action > 4 and action < 15
                if sparse_IS:
                    terrain_state = 1 if action > 9 else 2
                    p = 1 / (2)
                    terrain_num += 1
                else:
                    # 下一个生成地形类别
                    self.terrain_state = self.np_random.integers(1, self._STATES_)
                    self.weights.append(1)
                if debug: print(f"self.terrain_state = {self.terrain_state}")
            else:
                if action > 4 and action < 10:
                    action -= 5  # [0-4]
                elif action > 9:
                    action -= 10
                terrain_state = GRASS
                if sparse_IS:
                    terrain_counter = action + 5
                    p = 2 / TERRAIN_GRASS   # 10
                    terrain_num += 1
                else:
                    self.terrain_counter = self.np_random.integers(TERRAIN_GRASS / 2, TERRAIN_GRASS)
                    self.weights.append(1)
                if debug: print(f"self.terrain_counter = {self.terrain_counter}")
            terrain_oneshot = True

        if terrain_state == GRASS and not terrain_oneshot:
            terrain_velocity = 0.8 * terrain_velocity + \
                 0.01 * np.sign(TERRAIN_HEIGHT - current_y)
            if env_params is not None and env_params.altitude_fn is not None:
                self.current_y += self.terrain_velocity
                if i > TERRAIN_STARTPAD:
                    mid = TERRAIN_LENGTH * TERRAIN_STEP / 2.
                    x_ = (x - mid) * np.pi / mid
                    self.current_y = TERRAIN_HEIGHT + self.env_params.altitude_fn((x_, ))[0]
                    if i == TERRAIN_STARTPAD+1:
                        y_norm = self.env_params.altitude_fn((x_, ))[0]
                    self.current_y -= y_norm
                if debug: print(f"cppn used, i = {self.current_i}, y = {self.current_y}")
            else:
                if sparse_IS:
                    if action >=0 and action < 5:
                        delta_v = (action - 2 ) / 2
                    elif action >=5 and action < 15:
                        delta_v = (action - 9.5) / 4.5
                    terrain_velocity += delta_v  / SCALE
                else:
                    delta_v = self.np_random.uniform(-1, 1) / SCALE
                    self.terrain_velocity += delta_v
                    self.weights.append(1)
                    
                current_y += config.ground_roughness * terrain_velocity
                if debug: print(f"y = {round(self.current_y, 2)}")

        elif terrain_state == PIT and terrain_oneshot:
            if sparse_IS:
                pit_gap = 1.0 + (action - 5) * 0.2
                #p = 1 / (self.config.pit_gap[1] - self.config.pit_gap[0])
                #q = p
                #print('terrain counter == 0 and terrain_state == git, pit_gap=', pit_gap)
            else:
                pit_gap = 1.0 + self.np_random.uniform(*self.config.pit_gap)
                self.weights.append(1)
            if debug: print(f"pit_gap = {pit_gap}")

            terrain_counter = np.ceil(pit_gap)
            pit_diff = terrain_counter - pit_gap

            poly = [
                (x,              current_y),
                (x + TERRAIN_STEP, current_y),
                (x + TERRAIN_STEP, current_y - 4 * TERRAIN_STEP),
                (x,              current_y - 4 * TERRAIN_STEP),
            ]
            
            terrain_counter += 2
            env_poly= [(po[0] + TERRAIN_STEP * pit_gap, po[1]) for po in poly]
            cur_step_poly = env_poly
            original_y = current_y

        elif terrain_state == PIT and not terrain_oneshot:
            current_y = original_y
            if terrain_counter > 1:
                current_y -= 4 * TERRAIN_STEP
            if terrain_counter == 1:
                terrain_x[-1] = terrain_x[-1] - pit_diff * TERRAIN_STEP
                pit_diff = 0

        elif terrain_state == STUMP and terrain_oneshot:
            if sparse_IS:
                stump_width = 1
                stump_height = (action - 9) * 0.1
                if stump_height < 0:
                    stump_height = -stump_height
                stump_float = 0
                #p = 1
                #q = p
                #print('terrain counter == 0 and terrain_state == stump', 'height=',stump_height,'width=',stump_width,'float=',stump_float)
                
            else:
                stump_width = self.np_random.integers(*self.config.stump_width)
                stump_height = self.np_random.uniform(*self.config.stump_height)
                stump_float = self.np_random.integers(*self.config.stump_float)
                self.weights.append(1)
            
            if debug:
                print(f"stump_width = {stump_width}, stump_height = {stump_height}, stump_float = {stump_float}")
            
            terrain_counter = stump_width
            countery = stump_height
            poly = [
                (x,                      current_y + stump_float * TERRAIN_STEP),
                (x + stump_width * TERRAIN_STEP, current_y + stump_float * TERRAIN_STEP),
                (x + stump_width * TERRAIN_STEP, current_y + countery *
                 TERRAIN_STEP + stump_float * TERRAIN_STEP),
                (x,                     current_y + countery *
                 TERRAIN_STEP + stump_float * TERRAIN_STEP),
            ]
            
            cur_step_poly = poly
        
        
        terrain_oneshot = False
        terrain_counter -= 1
        

    if not cur_step_poly:
        cur_step_poly = [(init_i*TERRAIN_STEP, current_y),(current_i * TERRAIN_STEP, current_y),
                         (init_i*TERRAIN_STEP, current_y),(current_i * TERRAIN_STEP, current_y)]

    # print('cur_step_poly',cur_step_poly)
    return 0.1, cur_step_poly,terrain_num


"""
