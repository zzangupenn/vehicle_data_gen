import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import json
import os, sys
from tqdm import tqdm
import matplotlib.pyplot as plt

SEGMENT_LENGTH = 10
STEERING_LENGTH = 21e2
RESET_STEP = 210
VEL_SAMPLE_UP = 0.1
DENSITY_CURB = 0
STEERING_DENSITY = 4
RENDER = False
# SAVE_DIR = '/media/DATA/tuner_inn/sim_random_more/'
SAVE_DIR = '/workspace/data/tuner/sim_random/'

with open('maps/config_example_map.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)

def get_steers(sample_length, segment_length=10, peak_num=200):
    length = int(sample_length // segment_length)

    x = np.linspace(0, 1, length)
    y = np.zeros_like(x)

    for _ in range(peak_num):
        amplitude = np.random.rand() 
        frequency = np.random.randint(1, peak_num)
        phase = np.random.rand() * 2 * np.pi 

        y += amplitude * np.sin(2 * np.pi * frequency * x + phase)

    y -= np.mean(y)
    y_lower = np.min(y)
    z = y - y_lower
    y_upper = np.max(z)
    z = z/y_upper
    z = z*2
    z = z - 1.
    return z * 0.5

def curb_dense_points(samples, density=0.01):
    del_list = []
    for ind, sample in enumerate(samples):
        if ind == 0:
            pre_sample = sample
        else:
            if np.abs(sample - pre_sample) < density:
                del_list.append(ind)
            else:
                pre_sample = sample
    return np.delete(samples, del_list)


def warm_up(env, vel, warm_up_steps):
    # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
    
    obs, step_reward, done, info = env.reset(
        np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))

    step_count = 0
    while (step_count < warm_up_steps) and (np.abs(obs['x3'][0] - vel) > 0.01):
        try:
            obs, step_reward, done, info = env.step(np.array([[0.0, vel]]))
            step_count += 1
        except ZeroDivisionError:
            print('error warmup: ', step_count)




frictions = [0.5, 0.8, 1.1]
# frictions = [0.8]

if len(sys.argv) > 1:
    start_vel = float(sys.argv[1])
# start_vel = 8.0
print('start_vel', start_vel, 'end_vel', start_vel+VEL_SAMPLE_UP)
print('frictions', frictions)

def main():
    """
    main entry point
    """
    
    
    for friction in frictions: 
        print('friction', friction)
        total_controls = []
        total_states = []
        
        start = time.time()

        states = []
        controls = []
        steers = get_steers(STEERING_LENGTH * SEGMENT_LENGTH, SEGMENT_LENGTH, int(STEERING_LENGTH/100 * STEERING_DENSITY))
        if DENSITY_CURB != 0: steers = curb_dense_points(steers, DENSITY_CURB)
        # plt.plot(np.arange(steers.shape[0]), steers)
        # plt.show()

        step_count = 0
            
        # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
        env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                num_agents=1, timestep=0.01, model='MB', drive_control_mode='vel',
                steering_control_mode='angle')
        vel = np.random.uniform(start_vel, start_vel+VEL_SAMPLE_UP)
        warm_up(env, vel, 1000)
            
        with tqdm(total=len(steers)) as pbar:
            while step_count < len(steers):
                steer = steers[step_count]
                
                env.params['tire_p_dy1'] = friction  # mu_y
                env.params['tire_p_dx1'] = friction  # mu_x

                pbar.update(1)
                step_count += 1
                try:
                    for i in range(SEGMENT_LENGTH):
                        obs, rew, done, info = env.step(np.array([[steer, vel]]))
                        if RENDER: env.render(mode='human_fast')
                        
                    state = np.array([obs['x3'][0], obs['x4'][0], obs['x6'][0], obs['x11'][0]])
                    ## x3 = steering angle of front wheels
                    ## x4 = velocity in x-direction
                    ## x6 = yaw rate
                    ## x11 = velocity in y-direction
                    control = np.array([steer, vel])
                    states.append(state)
                    controls.append(control)
                    
                    if step_count % RESET_STEP == 0:
                        vel = np.random.uniform(start_vel, start_vel+VEL_SAMPLE_UP)
                        warm_up(env, vel, 1000)
                        if len(states) > 0:
                            # print(np.vstack(states).shape)
                            total_controls.append(np.vstack(controls))
                            total_states.append(np.vstack(states))
                            controls = []
                            states = []
                            
                except Exception as e:
                    print(e, ' at: ', step_count, ', reset to ', step_count//RESET_STEP * RESET_STEP)
                    step_count = step_count//RESET_STEP * RESET_STEP
                    pbar.n = step_count
                    pbar.refresh()
                    steers = get_steers(STEERING_LENGTH * SEGMENT_LENGTH, SEGMENT_LENGTH, int(STEERING_LENGTH/100 * STEERING_DENSITY))
                    if DENSITY_CURB != 0: steers = curb_dense_points(steers, DENSITY_CURB)
                    warm_up(env, vel, 1000)
                    controls = []
                    states = []
                    
                
                
                
        
        np.save(SAVE_DIR+'states_f{}_v{}_step{}.npy'.format(int(friction*10), 
                                                            int(np.round(start_vel, decimals=2)*100), 
                                                            RESET_STEP), np.stack(total_states))
        np.save(SAVE_DIR+'controls_f{}_v{}_step{}.npy'.format(int(friction*10), 
                                                                int(np.round(start_vel, decimals=2)*100), 
                                                                RESET_STEP), np.stack(total_controls))

        print('Real elapsed time:', time.time() - start)

if __name__ == '__main__':
    main()
