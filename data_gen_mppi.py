import time
import yaml
import gymnasium as gym
import numpy as np
from argparse import Namespace
import json
import os, sys
os.environ['F110GYM_PLOT_SCALE'] = str(5.)
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import Logger
from planner import pid
from utils.mb_model_params import params_real
from scipy.stats import truncnorm
from utils.utils import utilitySuite
# from moviepy.editor import ImageSequenceClip
# import pyglet
# pyglet.options['search_local_libs'] = True

import matplotlib.pyplot as plt
import matplotlib.animation as animation

NOISE = [0, 0, 0] # cxntrol_vel, control_steering, state 
EXP_NAME = 'kine_rand_uniform'
GYM_MODEL = "dynamic_ST"
INTEGRATION_DT = 0.1
STEERING_LENGTH = 21e2 * 1
RESET_STEP = 210
VEL_SAMPLE_UP = 3.0
DENSITY_CURB = 0
STEERING_PEAK_DENSITY = 1
RENDER = 0
PLOT = False
ACC_VS_CONTROL = False # THIS IS TRUE BEFORE, IDK IF IT IS STILL TRUE
SAVE_DIR = '/home/mu/workspace/data/' + EXP_NAME + '/'
PEAK_NUM = 10
# PEAK_NUM = int(STEERING_LENGTH/100 * STEERING_PEAK_DENSITY)

with open('maps/config_example_map.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)

def get_steers(sample_length, params, segment_length=1, peak_num=200):
    """
    Generate a random steering signal using a sum of sine waves.
    This function creates a synthetic steering signal by summing multiple sine waves with random amplitudes,
    frequencies, and phases. The resulting signal is normalized to the range [-1, 1], scaled by the maximum
    steering value specified in `params`, and clipped to the minimum and maximum steering limits.
    Args:
        sample_length (int): The total length of the steering signal to generate.
        params (dict): Dictionary containing steering parameters:
            - 's_max' (float): Maximum steering value.
            - 's_min' (float): Minimum steering value.
        segment_length (int, optional): The length of each segment in the signal. Defaults to 1.
        peak_num (int, optional): The number of sine wave components (peaks) to sum. Defaults to 200.
    Returns:
        np.ndarray: The generated steering signal as a 1D numpy array of length `sample_length // segment_length`.
    # This function synthesizes a random, smooth steering profile by combining multiple sine waves,
    # then normalizes and scales the result to fit within specified steering limits.
    """
    
    length = int(sample_length // segment_length)

    x = np.linspace(0, 1, length)
    y = np.zeros_like(x)

    for _ in range(int(peak_num)):
        amplitude = np.random.rand() 
        frequency = np.random.randint(1, peak_num)
        phase = np.random.rand() * 2 * np.pi 

        y += amplitude * np.sin(2 * np.pi * frequency * x + phase)

    y = y - np.min(y)
    y = y / np.max(y)
    y = y * 2 - 1 # scale to -1 to 1
    
    # rand_steer = truncnorm.rvs(-4.0, 4.0, size=1)[0] * 0.1
    # y += rand_steer
    y = y * params['s_max']
    y = np.clip(y, params['s_min'], params['s_max'])
    # plt.plot(np.arange(y.shape[0]), y)
    # plt.show()
    return y

def curb_dense_points(samples, density=0.01):
    """
    Removes points from the input array `samples` that are closer together than the specified `density` threshold.
    Iterates through the `samples` array and deletes any sample that is less than `density` distance away from the previous retained sample. 
    This effectively "thins out" densely packed points, ensuring that no two consecutive points in the returned array are closer than `density`.
    Args:
        samples (np.ndarray): 1D array of sample points (e.g., positions along a curb).
        density (float, optional): Minimum allowed distance between consecutive points. Defaults to 0.01.
    Returns:
        np.ndarray: Array of samples with densely packed points removed.
    # This function is useful for reducing the number of points in a dataset where points are too close together, 
    # such as when generating or processing curb or path data for vehicles.
    """
    
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

def get_obs_vel(obs):
    """
    Get the velocity from the observation
    :param obs: observation dictionary
    :return: velocity
    """
    vx = obs["linear_vels_x"][0]
    vy = obs["linear_vels_y"][0]
    return np.sqrt(vx**2 + vy**2)

def get_state(env, obs):
    """
    Get the state from the observation
    :param env: environment
    :param obs: observation dictionary
    :return: state vector
    """
    # State vector format:
    # [x position, y position, yaw angle, steering angle, velocity, yaw rate, slip angle]
    
    state = np.zeros((1, 7))
    state[0, 0] = obs['poses_x'][0]  # x position
    state[0, 1] = obs['poses_y'][0]  # y position
    state[0, 2] = obs['poses_theta'][0]  # yaw angle
    state[0, 3] = env.render_obs["steering_angles"][0]  # steering angle
    state[0, 4] = obs['linear_vels_x'][0]  # velocity
    state[0, 5] = obs['ang_vels_z'][0]  # yaw rate
    state[0, 6] = np.arctan2(obs['linear_vels_y'][0], obs['linear_vels_x'][0])  # slip angle
    return state

def warm_up(env, vel, warm_up_steps):
    """
    Gradually accelerates the environment's vehicle to a target velocity.
    This function resets the simulation environment to an initial pose and then
    repeatedly applies control inputs to the vehicle until its longitudinal velocity
    (`linear_vels_x`) is within 0.5 units of the desired target velocity (`vel`).
    The acceleration is computed based on the difference between the current and target velocities.
    The function returns the final observation and the environment.
    Args:
        env: The simulation environment object, which must implement `reset` and `step` methods.
        vel (float): The target velocity to reach during the warm-up phase.
        warm_up_steps (int): The maximum number of warm-up steps (currently unused in the function).
    Returns:
        obs (dict): The final observation from the environment after warm-up.
        env: The environment object, potentially updated after the warm-up process.
    """
    init_pose = np.zeros((1, 3))

    obs, _ = env.reset(
        # np.array([[0.0, 0.0, 0.0, 0.0, vel/1.1, 0.0, 0.0]])
        options={
            "poses": init_pose
        }
    )

    step_count = 0
    while (np.abs(obs['linear_vels_x'][0] - vel) > 0.5):
        try:
            accel = (vel - obs['linear_vels_x'][0]) * 0.89
            u_1 = obs['linear_vels_x'][0] + accel
            obs, _, _, _, _ = env.step(np.array([[0, u_1]]))
            # print(, obs['linear_vels_y'][0], get_obs_vel(obs), vel)
            step_count += 1
            # print('warmup step: ', step_count, 'error', get_obs_vel(obs), vel)
        except ZeroDivisionError:
            print('error warmup: ', step_count)
    print('warmup step: ', step_count, 'error', get_obs_vel(obs), vel)
    return obs, env

def plot_sanity_check(total_states):
    """
    Plots a sanity check for the total states.
    :param total_states: total states
    """
    rand_num = np.random.uniform(0, 10)
    print("Random number from 0 to 10:", rand_num)
    
    x = total_states[rand_num, :, 0]
    y = total_states[rand_num, :, 1]

    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'b-', lw=2)
    point, = ax.plot([], [], 'ro')
    ax.set_xlim(min(x) - 1, max(x) + 1)
    ax.set_ylim(min(y) - 1, max(y) + 1)
    ax.set_aspect('equal')
    ax.grid(True)

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def update(frame):
        line.set_data(x[:frame], y[:frame])
        point.set_data(x[frame], y[frame])
        return line, point

    ani = animation.FuncAnimation(fig, update, frames=len(x),
                                init_func=init, blit=True, interval=50)
    plt.show()


# frictions = [0.5, 0.8, 1.1]
frictions = [1.0]

if len(sys.argv) > 1:
    start_vel = float(sys.argv[1])
# start_vel = 8.0
print('start_vel', start_vel, 'end_vel', start_vel+VEL_SAMPLE_UP)
print('frictions', frictions)

def main():
    """
    main entry point
    """
    us = utilitySuite()
    logger = Logger(SAVE_DIR, EXP_NAME)
    logger.write_file(__file__)
    
    for friction in frictions: 
        print('friction', friction)
        total_controls = []
        total_states = []
        
        start = time.time()

        states = []
        controls = []
        steers = get_steers(RESET_STEP, params_real, peak_num=PEAK_NUM)
        if DENSITY_CURB != 0: steers = curb_dense_points(steers, DENSITY_CURB)
        # plt.plot(np.arange(steers.shape[0]), steers)
        # plt.show()
        
        step_count = 0
        steering_count = 0
            
        # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
        if ACC_VS_CONTROL:
            env = gym.make(
                'f1tenth_gym:f1tenth-v0', 
                map=conf.map_path, 
                map_ext=conf.map_ext,
                num_agents=1, 
                timestep=INTEGRATION_DT, 
                model=GYM_MODEL, 
                drive_control_mode='acc',
                steering_control_mode='vel'
            )
        else:
            env = gym.make(
                'f1tenth_gym:f1tenth-v0', 
                # map=conf.map_path, 
                # map_ext=conf.map_ext,
                # num_agents=1, 
                # timestep=INTEGRATION_DT, 
                # # reset_fn=reset_fn,
                # model=GYM_MODEL, 
                # drive_control_mode='vel',
                # steering_control_mode='angle'
            )
            default_config = env.default_config()
            default_config['map'] = conf.map_path
            default_config['timestep'] = INTEGRATION_DT
            default_config['num_agents'] = 1
            env.configure(
                config=default_config,
            )

        
        # vel = np.random.uniform(start_vel-VEL_SAMPLE_UP/2, start_vel+VEL_SAMPLE_UP/2)
        
        vel = start_vel + np.random.uniform(-VEL_SAMPLE_UP/2, VEL_SAMPLE_UP/2)
        print('here1')
        obs, env = warm_up(env, vel, 10000)
        print('here')
        with tqdm(total=STEERING_LENGTH) as pbar:
            while step_count < STEERING_LENGTH:
                if step_count % 42 == 0 and (step_count != 0) and (step_count % RESET_STEP != 0):
                    vel = start_vel + np.random.uniform(-VEL_SAMPLE_UP/2, VEL_SAMPLE_UP/2)
                steer = steers[steering_count]
                
                if ACC_VS_CONTROL:
                    # steering angle velocity input to steering velocity acceleration input
                    # v_combined = np.sqrt(obs['x4'][0] ** 2 + obs['x11'][0] ** 2)
                    state_st_0 = np.asarray(obs['state'][0])
                    v_combined = state_st_0[3]
                    
                    accl, sv = pid(vel, steer, v_combined, state_st_0[2], params_real['sv_max'], params_real['a_max'],
                                params_real['v_max'], params_real['v_min'])
                    accl += truncnorm.rvs(-params_real['a_max'], params_real['a_max'], size=1)[0] * 1
                    sv += truncnorm.rvs(-params_real['sv_max'], params_real['sv_max'], size=1)[0] * 1
                    accl = np.clip(accl, -params_real['a_max'], params_real['a_max'])
                    sv = np.clip(sv, -params_real['sv_max'], params_real['sv_max'])
                    control = np.array([sv, accl])
                else:
                    control = np.array([steer, vel])

                pbar.update(1)
                step_count += 1
                steering_count += 1
                try:
                    # Render the environment
                    # env.render(mode='human')
                    # frames.append()
                    
                    # Get the state and control
                    state_st_1 = get_state(env, obs)
                    states.append(state_st_1 + np.random.normal(scale=NOISE[2], size=state_st_1.shape))
                    controls.append(control)
                    
                    # Step the environment
                    obs, reward, terminated, truncated, info = env.step(np.array([[control[0] + np.random.normal(scale=NOISE[0]),
                                                                control[1] + np.random.normal(scale=NOISE[1])]]))
                        
                    # if RENDER: env.render(mode='human_fast')                    
                    
                    
                    if step_count % RESET_STEP == 0:
                        
                        steering_count = 0
                        steers = get_steers(RESET_STEP, params_real, peak_num=PEAK_NUM)
                        vel = start_vel + np.random.uniform(-VEL_SAMPLE_UP/2, VEL_SAMPLE_UP/2)
                        obs, env = warm_up(env, vel, 10000)
                        if len(states) > 0:
                            total_controls.append(np.vstack(controls))
                            total_states.append(np.vstack(states))
                            controls = []
                            states = []
                            
                except Exception as e:
                    steers = get_steers(RESET_STEP, params_real, peak_num=PEAK_NUM)
                    print(e, ' at: ', step_count, ', reset to ', step_count//RESET_STEP * RESET_STEP)
                    step_count = step_count//RESET_STEP * RESET_STEP
                    steering_count = 0
                    pbar.n = step_count
                    pbar.refresh()
                    steers = get_steers(RESET_STEP, params_real, peak_num=PEAK_NUM)
                    if DENSITY_CURB != 0: steers = curb_dense_points(steers, DENSITY_CURB)
                    obs, env = warm_up(env, vel, 10000)
                    controls = []
                    states = []
                    
            
        total_controls = np.asarray(total_controls)
        total_states = np.asarray(total_states)  
        print(total_controls.shape, total_states.shape)
        np.save(SAVE_DIR+'states_f{}_v{}.npy'.format(int(np.rint(friction*10)), 
                                                            int(np.rint(start_vel*100))), total_states)
        np.save(SAVE_DIR+'controls_f{}_v{}.npy'.format(int(np.rint(friction*10)), 
                                                                int(np.rint(start_vel*100))), total_controls)
        
        # axs = us.plt.get_fig([7, 1])
        # for ind in range(7):
        #     axs[ind].plot(np.arange(np.concatenate(total_states, axis=0).shape[0]), np.concatenate(total_states, axis=0)[:, ind], '.')        
        # us.plt.show_pause()

        if PLOT:
            plot_sanity_check(total_states)
        
        env.close()
        print('Real elapsed time:', time.time() - start, 'states_f{}_v{}.npy'.format(int(np.rint(friction*10)), 
                                                            int(np.rint(start_vel*100))))

if __name__ == '__main__':
    main()
