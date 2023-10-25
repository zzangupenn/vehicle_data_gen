import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import os, sys
from planner import PurePursuitPlanner, get_render_callback, pid
from utils.mb_model_params import param1


SEGMENT_LENGTH = 10
RENDER = True
SAVE_DIR = '/workspace/data/tuner/sim_random/'
MAP_DIR = './f1tenth_racetracks/'
ACC_VS_CONTROL = False

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

def warm_up(env, vel, warm_up_steps):
    # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
    
    obs, step_reward, done, info = env.reset(
        np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))

    step_count = 0
    while (step_count < warm_up_steps) and (np.abs(obs['x3'][0] - vel) > 0.01):
        try:
            obs, step_reward, done, info = env.step(np.array([[0.0, vel]]))
            if RENDER: 
                env.render(mode='human_fast')
                print(f'x {obs["x1"][0]:.2f}, y {obs["x2"][0]:.2f}, yaw {obs["x5"][0]:.2f}, yawrate {obs["x6"][0]:.2f}' + \
                        f', vx {obs["x4"][0]:.2f}, vy {obs["x11"][0]:.2f}, steer {obs["x3"][0]:.2f}')
            step_count += 1
        except ZeroDivisionError:
            print('error warmup: ', step_count)
            

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
    
    # NOTE: initialized states for forward
    if conf.wpt_thind == -1:
        init_theta = np.arctan2(waypoints[1, conf.wpt_yind] - waypoints[0, conf.wpt_yind], 
                                waypoints[1, conf.wpt_xind] - waypoints[0, conf.wpt_xind])
    else:
        init_theta = waypoints[0, conf.wpt_thind]
    
    return waypoints, conf, init_theta


def main():
    """
    main entry point
    """
    with open('maps/config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    
    frictions_ = [0.5]
    vels_ = [10.0]
    print('vels_', vels_)

    # for map_ind in range(7, 40):
    map_ind = 39
    map_info = np.genfromtxt('maps/map_info.txt', delimiter='|', dtype='str')[map_ind][1:]
    waypoints, conf, init_theta = load_map(MAP_DIR, map_info, conf, scale=10, reverse=True)
    print(map_ind, map_info[0])
    for vel in vels_:
        for friction in frictions_:
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
            if RENDER: env.add_render_callback(get_render_callback(planner))

            # # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
            obs, step_reward, done, info = env.reset(np.array([[waypoints[0, conf.wpt_xind], 
                                                                waypoints[0, conf.wpt_yind], 
                                                                init_theta, 0.0, 0.0, 0.0, 0.0]]))

            laptime = 0.0
            start = time.time()            
            controls = []
            states = []
            # warm_up(env, vel, 1000)
            while not done:
                speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'],
                                            work['vgain'])
                env.params['tire_p_dy1'] = friction  # mu_y
                env.params['tire_p_dx1'] = friction  # mu_x
                
                if ACC_VS_CONTROL:
                    # steering angle velocity input to steering velocity acceleration input
                    accl, sv = pid(speed, steer, obs['x4'][0], obs['x3'][0], param1['sv_max'], param1['a_max'],
                                param1['v_max'], param1['v_min'])
                    control = np.array([accl, sv])
                else:
                    control = np.array([steer, vel])
                    
                for i in range(SEGMENT_LENGTH):
                    obs, rew, done, info = env.step(np.array([[control[0], control[1]]]))
                    step_reward += rew

                state = np.array([obs['x3'][0], obs['x4'][0], obs['x6'][0], obs['x11'][0]])
                ## x3 = steering angle of front wheels
                ## x4 = velocity in x-direction
                ## x6 = yaw rate
                ## x11 = velocity in y-direction
                
                states.append(state)
                controls.append(control)

                laptime += step_reward
                if RENDER: 
                    env.render(mode='human_fast')
                    print(f'x {obs["x1"][0]:.2f}, y {obs["x2"][0]:.2f}, yaw {obs["x5"][0]:.2f}, yawrate {obs["x6"][0]:.2f}' + \
                        f', vx {obs["x4"][0]:.2f}, vy {obs["x11"][0]:.2f}, steer {obs["x3"][0]:.2f}')

            np.save(SAVE_DIR + 'states_f{}_v{}.npy'.format(int(np.rint(friction*10)), int(np.rint(vel*100))), np.vstack(states))
            np.save(SAVE_DIR + 'controls_f{}_v{}.npy'.format(int(np.rint(friction*10)), int(np.rint(vel*100))), np.vstack(controls))

            print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)


if __name__ == '__main__':
    main()


