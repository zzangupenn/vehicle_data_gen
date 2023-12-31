import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import os, sys
from planner import PurePursuitPlanner, get_render_callback, pid
from utils.mb_model_params import param1
from additional_renderers import *


SEGMENT_LENGTH = 20
RENDER = True
SAVE_DIR = '/home/lucerna/Documents/DATA/tuner_inn/track39/'
MAP_DIR = './f1tenth_racetracks/'
ACC_VS_CONTROL = True
VEL_SAMPLE_UP = 1.0
SAVE_STEP = 210
        
    
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
    
    # NOTE: initialized states for forward
    if conf.wpt_thind == -1:
        init_theta = np.arctan2(waypoints[1, conf.wpt_yind] - waypoints[0, conf.wpt_yind], 
                                waypoints[1, conf.wpt_xind] - waypoints[0, conf.wpt_xind])
    else:
        init_theta = waypoints[0, conf.wpt_thind]
    
    return waypoints, conf, init_theta

def state_mb2nf(mb_state):
    return np.array([mb_state[0], mb_state[1], mb_state[2],
                    mb_state[3], mb_state[4], mb_state[5],
                    mb_state[10]])


if len(sys.argv) > 1:
    start_vel = float(sys.argv[1])
    # vels = [vel]
    vels = np.arange(start_vel, start_vel + 6, 1.0)

def main():
    """
    main entry point
    """
    with open('maps/config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    
    frictions = [0.5]
    # vels = np.arange(8, 9, 1)
    # print('vels', vels)

    # for map_ind in range(7, 40):
    map_ind = 39
    
    for friction in frictions:
        total_states = []
        total_controls = []
        for vel in vels:
            for reverse in range(2):
                map_info = np.genfromtxt('maps/map_info.txt', delimiter='|', dtype='str')[map_ind][1:]
                print(map_ind, map_info[0], 'reverse', reverse)
                waypoints, conf, init_theta = load_map(MAP_DIR, map_info, conf, scale=start_vel-2, reverse=reverse)
                
                print('vel', vel)
                print('friction', friction)

                work = {'mass': 1225.88, 'lf': 0.80597534362552312, 'tlad': 10.6461887897713965, 'vgain': 0.950338203837889}
                planner = PurePursuitPlanner(conf, 0.805975 + 1.50876)
                planner.waypoints = waypoints
                
                if ACC_VS_CONTROL:
                    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                            num_agents=1, timestep=0.01, model='MB', drive_control_mode='acc',
                            steering_control_mode='vel')
                else:
                    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                                num_agents=1, timestep=0.01, model='MB', drive_control_mode='vel',
                                steering_control_mode='angle')
                    
                map_waypoint_renderer = MapWaypointRenderer(waypoints)
                renderers = [map_waypoint_renderer]
                if RENDER: env.add_render_callback(get_render_callback(renderers))

                # # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
                obs, step_reward, done, info = env.reset(np.array([[waypoints[0, conf.wpt_xind], 
                                                                    waypoints[0, conf.wpt_yind], 
                                                                    init_theta, 0.0, start_vel, 0.0, 0.0]]))

                laptime = 0.0
                start = time.time()            
                controls = []
                states = []
                cnt = 0
                while not done:
                    if cnt % 42 == 0:
                        target_vel = vel + np.random.uniform(-VEL_SAMPLE_UP/2, VEL_SAMPLE_UP/2)
                    
                    speed, steer, ind = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'],
                                                work['vgain'], target_vel)
                    env.params['tire_p_dy1'] = friction  # mu_y
                    env.params['tire_p_dx1'] = friction  # mu_x
                    
                    if ACC_VS_CONTROL:
                        # steering angle velocity input to steering velocity acceleration input
                        accl, sv = pid(target_vel, steer, obs['x4'][0], obs['x3'][0], param1['sv_max'], param1['a_max'],
                                    param1['v_max'], param1['v_min'])
                        control = np.array([sv, accl])
                    else:
                        control = np.array([steer, vel])
                    
                        
                    for i in range(SEGMENT_LENGTH):
                        obs, rew, done, info = env.step(np.array([[control[0], control[1]]]))
                        step_reward += rew
                    

                    state = obs['state'][0]
                    ## x3 = steering angle of front wheels
                    ## x4 = velocity in x-direction
                    ## x6 = yaw rate
                    ## x11 = velocity in y-direction
                    
                    cnt += 1
                    states.append(state[[2, 3, 5, 10]])
                    controls.append(control)
                    
                    if cnt % SAVE_STEP == 0:
                        total_states.append(np.stack(states))
                        total_controls.append(np.stack(controls))
                        controls = []
                        states = []

                    laptime += step_reward
                    if RENDER: 
                        map_waypoint_renderer.update(state[:2])
                        env.render(mode='human_fast')
                        
                print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)
                print(np.asarray(total_states).shape)
                print(np.asarray(total_controls).shape)
            np.save(SAVE_DIR + 'states_f{}_v{}.npy'.format(int(np.rint(friction*10)), int(np.rint(vel*100))), total_states)
            np.save(SAVE_DIR + 'controls_f{}_v{}.npy'.format(int(np.rint(friction*10)), int(np.rint(vel*100))), total_controls)

            


if __name__ == '__main__':
    main()


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