import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import os, sys
from planner import PurePursuitPlanner, get_render_callback
from utils import frenet_to_cartesian, cartesian_to_frenet
import matplotlib.pyplot as plt

SEGMENT_LENGTH = 10
RENDER = True
SAVE_DIR = '/workspace/data/tuner/sim_random/'
MAP_DIR = './f1tenth_racetracks/'

# maps = os.listdir(MAP_DIR)[:-1]
# del maps[3]
# print(maps)
# row = '# wpt_path|wpt_delim|wpt_rowskip|wpt_xind|wpt_yind|wpt_thind|wpt_vind'
# file1 = open("map_info.txt", "w")
# file1.write(row + '\n')
# for ind in range(len(maps)):
#     file1.write(str(ind*2) + '|' + maps[ind] + '/' + maps[ind] + '_centerline.csv|,|1|0|1|-1|-1' + '\n')
#     file1.write(str(ind*2+1) + '|' + maps[ind] + '/' + maps[ind] + '_raceline.csv|;|3|1|2|3|5' + '\n')

# exit()

if len(sys.argv) > 1:
    start_vel = float(sys.argv[1])
    # vels_ = [vel]
    vels_ = np.arange(start_vel, start_vel + 1, 0.2)
    
def load_map(MAP_DIR, map_info, conf, scale=1, reverse=False):
    """
    loads waypoints
    """
    conf.wpt_path = map_info[0]
    conf.wpt_delim = map_info[1]
    conf.wpt_rowskip = int(map_info[2])
    conf.wpt_xind = int(map_info[3])
    conf.wpt_yind = int(map_info[4])
    conf.wpt_thind = int(map_info[5])
    conf.wpt_vind = int(map_info[6])
    
    waypoints = np.loadtxt(MAP_DIR + conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
    if reverse: # NOTE: reverse map
        waypoints = waypoints[::-1]
        waypoints[:, conf.wpt_thind] = waypoints[:, conf.wpt_thind] + 3.14
    waypoints[:, conf.wpt_yind] = waypoints[:, conf.wpt_yind] * scale
    waypoints[:, conf.wpt_xind] = waypoints[:, conf.wpt_xind] * scale # NOTE: map scales
    return waypoints, conf


def friction_func(pose_frenet, waypoints):
    s_max = np.max(waypoints[:, 0]) # Handles the case when waypoints is flipped
    s = pose_frenet[0]
    ey = pose_frenet[1]
    ey_max = 10 # Maximum lateral error - Track width
    if abs(ey) > ey_max:
        return np.nan
    if s < 0.5 * s_max:
        # Linear change from 1.1 abs(ey) = 0 to 0.5 abs(ey) >= ey_max
        ey = min(abs(ey), ey_max)
        return 1.1 - 0.6 * ey / ey_max
    else:
        ey = min(abs(ey), ey_max)
        return 0.5 - 0.3 * ey / ey_max

def test_friction_func():
    with open('maps/config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    map_ind = 11    
    map_info = np.genfromtxt('maps/map_info.txt', delimiter='|', dtype='str')[map_ind][1:]
    waypoints, conf = load_map(MAP_DIR, map_info, conf, scale=10, reverse=True)
    # waypoints is flipped, so we need to flip it back
    waypoints = waypoints[::-1]

    # Sample 10,000 points over s, each with 100 points over ey
    s = np.linspace(0, np.max(waypoints[:, 0]), 10000)
    ey = np.linspace(-10, 10, 20)
    mu = np.zeros((s.shape[0], ey.shape[0]))
    for i in range(s.shape[0]):
        for j in range(ey.shape[0]):
            mu[i, j] = friction_func(np.array([s[i], ey[j], 0]), waypoints)
            
    # convert to cartesian
    x = np.zeros(mu.shape)
    y = np.zeros(mu.shape)
    for i in range(s.shape[0]):
        for j in range(ey.shape[0]):
            x[i, j], y[i, j], _ = frenet_to_cartesian(np.array([s[i], ey[j], 0]), waypoints)
    
    # Plot colorbar with x,y vs mu and the waypoints
    plt.figure()
    # plt.plot(waypoints[:, 1], waypoints[:, 2])
    plt.pcolor(x, y, mu)
    # plt.colorbar()
    plt.show()

    
def main():
    """
    main entry point
    """
    # Visualize friction function
    test_friction_func()

    with open('maps/config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    
    # frictions_ = [0.5, 0.8, 1.1]
    friction_funcs = [friction_func]
    vels_ = [11.0]
    print('vels_', vels_)

    # for map_ind in range(7, 40):
    map_ind = 39
    map_info = np.genfromtxt('maps/map_info.txt', delimiter='|', dtype='str')[map_ind][1:]
    waypoints, conf = load_map(MAP_DIR, map_info, conf, scale=10, reverse=True)
    print(map_ind, map_info[0])
    for vel in vels_:
        for friction_func_ in friction_funcs:
            print('vel', vel)
            print('friction_', friction_func_.__name__)

            work = {'mass': 1225.88, 'lf': 0.80597534362552312, 'tlad': 10.6461887897713965, 'vgain': 0.950338203837889}
            planner = PurePursuitPlanner(conf, 0.805975 + 1.50876)
            planner.waypoints = waypoints
            
            env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                        num_agents=1, timestep=0.02, model='MB', drive_control_mode='vel',
                        steering_control_mode='angle')
            env.add_render_callback(get_render_callback(planner))

            # # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
            # obs, step_reward, done, info = env.reset(
            #     np.array([[conf.sx, conf.sy, conf.stheta, 0.0, 0.0, 0.0, 0.0]]))

            # NOTE: initialized states for forward
            if conf.wpt_thind == -1:
                init_theta = np.arctan2(waypoints[1, conf.wpt_yind] - waypoints[0, conf.wpt_yind], 
                                        waypoints[1, conf.wpt_xind] - waypoints[0, conf.wpt_xind])
            else:
                init_theta = waypoints[0, conf.wpt_thind]
            obs, step_reward, done, info = env.reset(
                np.array([[waypoints[0, conf.wpt_xind], 
                        waypoints[0, conf.wpt_yind], 
                        init_theta, 0.0, vel, 0.0, 0.0]]))


            laptime = 0.0
            start = time.time()            
            controls = []
            states = []
            while not done:
                speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'],
                                            work['vgain'])
                pose_frenet = cartesian_to_frenet(np.array([obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]]), waypoints)

                env.params['tire_p_dy1'] = friction_func_(pose_frenet, waypoints)  # mu_y
                env.params['tire_p_dx1'] = friction_func_(pose_frenet, waypoints)  # mu_x

                step_reward = 0.0
                state = np.array([obs['x3'][0], obs['x4'][0], obs['x6'][0], obs['x11'][0]])
                # print('start', state)
                for i in range(SEGMENT_LENGTH):
                    obs, rew, done, info = env.step(np.array([[steer, vel]]))
                    step_reward += rew

                    # state = np.array([obs['poses_x'][0], obs['poses_y'][0], 
                    #         obs['poses_theta'][0], work['tlad'],
                    #         work['vgain']])
                state = np.array([obs['x3'][0], obs['x4'][0], obs['x6'][0], obs['x11'][0]])
                # print('end', state)
                ## x3 = steering angle of front wheels
                ## x4 = velocity in x-direction
                ## x6 = yaw rate
                ## x11 = velocity in y-direction
                control = np.array([steer, vel])
                states.append(state)
                controls.append(control)

                laptime += step_reward
                # NOTE: render
                if RENDER: env.render(mode='human_fast')

            # file_name_ = 'data/rounded/' # NOTE
            file_name_ = '/media/DATA/tuner_inn/sim_track/'
            # np.save(file_name_+'states_f{}_v{}.npy'.format(int(friction_*10), int(np.round(vel, decimals=1)*10)), np.vstack(states))
            # np.save(file_name_+'controls_f{}_v{}.npy'.format(int(friction_*10), int(np.round(vel, decimals=1)*10)), np.vstack(controls))

            print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)


if __name__ == '__main__':
    main()


