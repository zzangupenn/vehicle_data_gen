import time
import f1tenth_gym
import f1tenth_gym.envs
from requests import get
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
from f1tenth_gym.envs.f110_env import F110Env
from dotenv import load_dotenv
# from moviepy.editor import ImageSequenceClip
# import pyglet
# pyglet.options['search_local_libs'] = True

import matplotlib.pyplot as plt
import matplotlib.animation as animation

config = None
with open('config/config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

load_dotenv()  # automatically loads from .env in current dir
ws_home = os.getenv("MY_WS_HOME")

NOISE = config['simulation']['noise'] # control_vel, control_steering, state
EXP_NAME = config['experiment']['name']
GYM_MODEL = config['simulation']['gym_model']
INTEGRATION_DT = config['simulation']['integration_dt']
STEERING_LENGTH = config['simulation']['steering_length']
RESET_STEP = config['simulation']['reset_step']
VEL_SAMPLE_UP = float(config['simulation']['vel_sample_up'])
DENSITY_CURB = int(config['simulation']['density_curb'])
STEERING_PEAK_DENSITY = int(config['simulation']['steering_peak_density'])
RENDER = bool(config['experiment']['render'])
PLOT = bool(config['experiment']['plot'])
ACC_VS_CONTROL = bool(config['experiment']['acc_vs_control'])
# SAVE_DIR = ws_home + config['experiment']['save_dir']
SAVE_DIR = config['experiment']['save_dir']
PEAK_NUM = config['simulation']['peak_num']
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
    states = get_state(obs)
    vx = states[0, 3]  # x velocity
    return vx

def get_state(obs):
    """
    Get the state from the observation
    :param env: environment
    :param obs: observation dictionary
    :return: state vector
    """
    # State vector format:
    # [x position, y position, yaw angle, steering angle, velocity, yaw rate, slip angle]
    # dict_keys(['scan', 'std_state', 'state', 'collision', 'lap_time', 'lap_count', 'sim_time', 'frenet_pose'])
    state = np.asarray(obs['agent_0']['std_state'])
    state = state.reshape((1, 7))  # Ensure state is a 2D array with shape (1, 7)
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

    # [x, y, steering angle, velocity, yaw, yaw_rate, beta]
    obs, _ = env.reset(
        # np.array([[0.0, 0.0, 0.0, 0.0, vel/1.1, 0.0, 0.0]])
        options={
            # "poses": init_pose,
            "states": np.array([[0.0, 0.0, 0.0, vel/1.05, 0.0, 0.0, 0.0]]),
        }
    )

    # return obs, env

    # The following function is not used for latest gym
    step_count = 0
    state_v = 0
    while (abs(state_v - vel) > 0.5):
        try:
            accel = (vel - state_v) * 0.1
            u_1 = accel if ACC_VS_CONTROL else state_v + accel
            obs, _, _, _, _ = env.step(np.array([[0.0, u_1]]))
            state_v = get_obs_vel(obs)
            # print(, obs['linear_vels_y'][0], get_obs_vel(obs), vel)
            # print(step_count)
            step_count += 1
            # print('warmup step: ', step_count, 'error', get_obs_vel(obs), vel)
        except ZeroDivisionError:
            print('error warmup: ', step_count)
    # print('warmup step: ', step_count, 'error', get_obs_vel(obs), vel)
    return obs, env

def plot_sanity_check(total_states):
    """
    Plots a sanity check for the total states.
    :param total_states: total states
    """
    rand_num = int(np.random.uniform(0, 10))
    print("Random number from 0 to 10:", rand_num)
    
    x = total_states[rand_num, :, 0]
    y = total_states[rand_num, :, 1]

    # print('x', x.shape, 'y', y.shape)

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
        point.set_data([x[frame]], [y[frame]])
        return line, point

    ani = animation.FuncAnimation(fig, update, frames=len(x),
                                init_func=init, blit=True, interval=50)
    ani.save("sanity_check.gif", writer="pillow")
    # plt.show()


# frictions = [0.5, 0.8, 1.1]
# frictions = [1.0]
frictions = config['frictions']
# print('frictions', frictions)
# exit(0)

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
            
        # init vector = [x, y, steering angle, velocity, yaw, yaw_rate, beta]
        env_config = {
            "seed": 12345,
            "map": "maps/example_map",
            "map_scale": 1.0,
            "params": f1tenth_gym.envs.f110_env.F110Env.fullscale_vehicle_params(),
            "num_agents": 1,
            "timestep": 0.1,
            "integrator_timestep": 0.01,
            "ego_idx": 0,
            "max_laps": 'inf',  # 'inf' for infinite laps, or a positive integer
            "integrator": "rk4",
            "model": "st", # "ks", "st", "mb"
            "control_input": ["accl", "steering_speed"],
            "observation_config": {"type": "direct"},
            "reset_config": {"type": None},
            "enable_rendering": False,
            "enable_scan": False, # NOTE no lidar scan and collision if False
            "lidar_fov" : 4.712389,
            "lidar_num_beams": 1080,
            "lidar_range": 30.0,
            "lidar_noise_std": 0.01,
            "steer_delay_buffer_size": 0,
            "compute_frenet": False, 
            "collision_check_method": "bounding_box", # "lidar_scan", "bounding_box"
            "loop_counting_method": "frenet_based", # "toggle", "frenet_based", "winding_angle"
        }
        # env_config =
        if ACC_VS_CONTROL:
            env_config["control_input"] = ["accl", "steering_speed"]
            env = gym.make(
                'f1tenth_gym:f1tenth-v0',
                config=env_config, 
            )
        else:
            env_config["control_input"] = ["speed", "steering_speed"]
            env = gym.make(
                'f1tenth_gym:f1tenth-v0',
                config=env_config,
            )
        # print(env.config)
        
        # vel = np.random.uniform(start_vel-VEL_SAMPLE_UP/2, start_vel+VEL_SAMPLE_UP/2)
        
        vel = start_vel + np.random.uniform(-VEL_SAMPLE_UP/2, VEL_SAMPLE_UP/2)
        obs, env = warm_up(env, vel, 10000)
        with tqdm(total=STEERING_LENGTH) as pbar:
            while step_count < STEERING_LENGTH:
                if step_count % 42 == 0 and (step_count != 0) and (step_count % RESET_STEP != 0):
                    vel = start_vel + np.random.uniform(-VEL_SAMPLE_UP/2, VEL_SAMPLE_UP/2)
                steer = steers[steering_count]
                
                env.params['tire_p_dy1'] = friction  # mu_y
                env.params['tire_p_dx1'] = friction  # mu_x
                if ACC_VS_CONTROL:
                    # steering angle velocity input to steering velocity acceleration input
                    # v_combined = np.sqrt(obs['x4'][0] ** 2 + obs['x11'][0] ** 2)
                    state_st_0 = obs['agent_0']['std_state']
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
                    state_st_1 = get_state(obs)
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
        
        axs = us.plt.get_fig([7, 1])
        for ind in range(7):
            axs[ind].plot(np.arange(np.concatenate(total_states, axis=0).shape[0]), np.concatenate(total_states, axis=0)[:, ind], '.')        
        # us.plt.show_pause()

        if PLOT:
            plot_sanity_check(total_states)
        
        # env.close()
        print('Real elapsed time:', time.time() - start, 'states_f{}_v{}.npy'.format(int(np.rint(friction*10)), 
                                                            int(np.rint(start_vel*100))))

if __name__ == '__main__':
    main()
