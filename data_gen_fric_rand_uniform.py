import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import json
import os, sys
os.environ['F110GYM_PLOT_SCALE'] = str(5.)
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import Logger
from planner import pid
from utils.mb_model_params import param1
from scipy.stats import truncnorm

NOISE = [0, 0, 0] # control_vel, control_steering, state 

EXP_NAME = 'fric3_rand_acc2_t01'
DT = 0.1

INTEGRATION_DT = 0.002
SEGMENT_LENGTH = int(np.rint(DT/INTEGRATION_DT))
STEERING_LENGTH = 21e2 * 4
RESET_STEP = 210
VEL_SAMPLE_UP = 3.0
DENSITY_CURB = 0
STEERING_PEAK_DENSITY = 2.5
RENDER = False
ACC_VS_CONTROL = True
# SAVE_DIR = '/home/lucerna/Documents/DATA/tuner/' + EXP_NAME + '/'
SAVE_DIR = '/workspace/data/tuner/' + EXP_NAME + '/'
# SAVE_DIR = '/workspace/data/tuner/sim_random_noise/'

with open('maps/config_example_map.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)

def get_steers(sample_length, segment_length=10, peak_num=200):
    length = int(sample_length // segment_length)

    x = np.linspace(0, 1, length)
    y = np.zeros_like(x)

    for _ in range(int(peak_num)):
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
    # z = z * 0.8
    
    rand_steer = truncnorm.rvs(-4.0, 4.0, size=1)[0] * 0.1
    z += rand_steer
    z[np.where(z > 0.4)] = 0.4
    z[np.where(z < -0.4)] = -0.4
    return z

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
        np.array([[0.0, 0.0, 0.0, 0.0, vel/1.1, 0.0, 0.0]]))

    step_count = 0
    while (np.abs(obs['x4'][0] - vel) > 0.5):
        try:
            accel = (vel - obs['x4'][0]) * 0.1
            obs, step_reward, done, info = env.step(np.array([[0.0, accel]]))
            step_count += 1
        except ZeroDivisionError:
            print('error warmup: ', step_count)
    print('warmup step: ', step_count, 'error', obs['x4'][0], vel)
    return obs




# frictions = [0.5, 0.8, 1.1]
frictions = [0.5]

if len(sys.argv) > 1:
    start_vel = float(sys.argv[1])
# start_vel = 8.0
print('start_vel', start_vel, 'end_vel', start_vel+VEL_SAMPLE_UP)
print('frictions', frictions)

def main():
    """
    main entry point
    """
    logger = Logger(SAVE_DIR, EXP_NAME)
    logger.write_file(__file__)
    
    for friction in frictions: 
        print('friction', friction)
        total_controls = []
        total_states = []
        
        start = time.time()

        states = []
        controls = []
        steers = get_steers(RESET_STEP * SEGMENT_LENGTH, SEGMENT_LENGTH, int(STEERING_LENGTH/100 * STEERING_PEAK_DENSITY))
        if DENSITY_CURB != 0: steers = curb_dense_points(steers, DENSITY_CURB)
        # plt.plot(np.arange(steers.shape[0]), steers)
        # plt.show()

        step_count = 0
        steering_count = 0
            
        # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
        if ACC_VS_CONTROL:
            env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                    num_agents=1, timestep=INTEGRATION_DT, model='MB', drive_control_mode='acc',
                    steering_control_mode='vel')
        else:
            env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                        num_agents=1, timestep=INTEGRATION_DT, model='MB', drive_control_mode='vel',
                        steering_control_mode='angle')
        # vel = np.random.uniform(start_vel-VEL_SAMPLE_UP/2, start_vel+VEL_SAMPLE_UP/2)
        vel = start_vel + np.random.uniform(-VEL_SAMPLE_UP/2, VEL_SAMPLE_UP/2)
        obs = warm_up(env, vel, 10000)
        with tqdm(total=STEERING_LENGTH) as pbar:
            while step_count < STEERING_LENGTH:
                if step_count % 42 == 0 and (step_count != 0) and (step_count % RESET_STEP != 0):
                    vel = start_vel + np.random.uniform(-VEL_SAMPLE_UP/2, VEL_SAMPLE_UP/2)
                steer = steers[steering_count]
                
                env.params['tire_p_dy1'] = friction  # mu_y
                env.params['tire_p_dx1'] = friction  # mu_x
                
                if ACC_VS_CONTROL:
                    # steering angle velocity input to steering velocity acceleration input
                    v_combined = np.sqrt(obs['x4'][0] ** 2 + obs['x11'][0] ** 2)
                    accl, sv = pid(vel, steer, v_combined, obs['x3'][0], param1['sv_max'], param1['a_max'],
                                param1['v_max'], param1['v_min'])
                    control = np.array([sv, accl])
                else:
                    control = np.array([steer, vel])
                    
                # print(step_count, 'accl', accl, 'vel', vel, 'start_vel', start_vel, 'x4', obs['x4'][0], 'x11', obs['x11'][0])
                # if np.abs(accl) > 0.1:
                # print(step_count, 'accl', accl, 'vel', vel, 'v_combined', v_combined, 'x4', obs['x4'][0], 'x11', obs['x11'][0])
                # print('sv', sv, 'steer', steer, 'x3', obs['x3'][0])

                pbar.update(1)
                step_count += 1
                steering_count += 1
                try:
                    for i in range(SEGMENT_LENGTH):
                        obs, rew, done, info = env.step(np.array([[control[0] + np.random.normal(scale=NOISE[0]),
                                                                   control[1] + np.random.normal(scale=NOISE[1])]]))
                        if RENDER: env.render(mode='human_fast')
                        
                    state = np.array([obs['x3'][0], obs['x4'][0], obs['x6'][0], obs['x11'][0]])
                    ## x3 = steering angle of front wheels
                    ## x4 = velocity in x-direction
                    ## x6 = yaw rate
                    ## x11 = velocity in y-direction
                    # control = np.array([steer, vel])
                    states.append(state + np.random.normal(scale=NOISE[2], size=state.shape))
                    controls.append(control)
                    # print(control)
                    
                    if step_count % RESET_STEP == 0:
                        
                        steering_count = 0
                        steers = get_steers(RESET_STEP * SEGMENT_LENGTH, SEGMENT_LENGTH, int(STEERING_LENGTH/100 * STEERING_PEAK_DENSITY))
                        vel = start_vel + np.random.uniform(-VEL_SAMPLE_UP/2, VEL_SAMPLE_UP/2)
                        obs = warm_up(env, vel, 10000)
                        # print(step_count, 'reset', 'vel', vel, 'x4', obs['x4'][0], 'x11', obs['x11'][0])
                        if len(states) > 0:
                            # print(np.vstack(states).shape)
                            total_controls.append(np.vstack(controls))
                            total_states.append(np.vstack(states))
                            controls = []
                            states = []
                            
                except Exception as e:
                    steers = get_steers(RESET_STEP * SEGMENT_LENGTH, SEGMENT_LENGTH, int(STEERING_LENGTH/100 * STEERING_PEAK_DENSITY))
                    print(e, ' at: ', step_count, ', reset to ', step_count//RESET_STEP * RESET_STEP)
                    step_count = step_count//RESET_STEP * RESET_STEP
                    steering_count = 0
                    pbar.n = step_count
                    pbar.refresh()
                    steers = get_steers(STEERING_LENGTH * SEGMENT_LENGTH, SEGMENT_LENGTH, int(STEERING_LENGTH/100 * STEERING_PEAK_DENSITY))
                    if DENSITY_CURB != 0: steers = curb_dense_points(steers, DENSITY_CURB)
                    _ = warm_up(env, vel, 10000)
                    controls = []
                    states = []
                    
                
                
        # print(np.max(total_controls, axis=1), np.min(total_controls, axis=1))
        
        np.save(SAVE_DIR+'states_f{}_v{}.npy'.format(int(np.rint(friction*10)), 
                                                            int(np.rint(start_vel*100))), np.stack(total_states))
        np.save(SAVE_DIR+'controls_f{}_v{}.npy'.format(int(np.rint(friction*10)), 
                                                                int(np.rint(start_vel*100))), np.stack(total_controls))

        print('Real elapsed time:', time.time() - start, 'states_f{}_v{}.npy'.format(int(np.rint(friction*10)), 
                                                            int(np.rint(start_vel*100))))

if __name__ == '__main__':
    main()
