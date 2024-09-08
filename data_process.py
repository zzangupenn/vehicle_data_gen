import matplotlib.pyplot as plt
import numpy as np
from utils.utils import DataProcessor, ConfigYAML, Logger

TEST = 0
TRAIN_DATADIR = '/media/lucerna/DATA/kine_rand_uniform'
# TRAIN_DATADIR = '/home/lucerna/Documents/DATA/tuner_inn/track39'
if TEST:
    DATADIR = TRAIN_DATADIR + '_test/'
else:
    DATADIR = TRAIN_DATADIR + '/'
TRAIN_SEGMENT = 2
TIME_INTERVAL = 0.1
SAVE_NAME = ''

logger = Logger(DATADIR, SAVE_NAME)
logger.write_file(__file__)

# vlist = np.hstack([np.arange(0, 1, 0.1) + i for i in np.arange(5, 9)])
vlist = np.arange(5.0, 21.0, 1)
# flist = [0.5, 0.8, 1.1]
flist = [1.0]
print('vlist', vlist)
dp = DataProcessor()

all_friction_states = []
all_friction_control = []
for ind, friction_ in enumerate(flist):
    total_states = []
    total_controls = []
    for vel in vlist:
        filename = 'states_f{}_v{}.npy'.format(int(np.rint(friction_*10)),
                                                       int(np.rint(vel*100)))
        controls_filename = 'controls_f{}_v{}.npy'.format(int(np.rint(friction_*10)), 
                                                                  int(np.rint(vel*100)))
        
        states = np.load(DATADIR + filename)
        controls = np.load(DATADIR + controls_filename)
        total_states.append(states)
        total_controls.append(controls)

    all_friction_states.append(np.vstack(total_states))
    all_friction_control.append(np.vstack(total_controls))
    # all_friction_states.append(total_states)
    # all_friction_control.append(total_controls)

all_friction_control = np.asarray(all_friction_control)
all_friction_states = np.asarray(all_friction_states)[..., (2, 3, 5, 6)]
print('all_friction_states', all_friction_states.shape, np.isnan(all_friction_states).sum(), np.isinf(all_friction_states).sum())

## normalization
normalization_param = []
for ind in range(4):
    _, param = dp.data_normalize(np.vstack(np.vstack(all_friction_states))[:, ind])
    normalization_param.append(param)

dynamics = []
for ind, friction_ in enumerate(flist):
    states_fric = all_friction_states[ind]
    # controls_fric = all_friction_control[ind]
    for segment_ind in range(states_fric.shape[0]):
        states = states_fric[segment_ind]
        # controls = controls_fric[segment_ind]
        states = np.vstack([states[i:i+2][None, :] for i in range(0, len(states)-2+1, 2)])
        dynamics.append((states[:, 1, :] - states[:, 0, :]) / TIME_INTERVAL)
dynamics = np.asarray(dynamics)
print('dynamics', dynamics.shape, np.isnan(dynamics).sum(), np.isinf(dynamics).sum())


dynamics = np.vstack(dynamics)
for ind in range(4):
    _, param = dp.data_normalize(dynamics[:, ind])
    normalization_param.append(param)
    
for ind in range(2):
    _, param = dp.data_normalize(np.vstack(np.vstack(all_friction_control))[:, ind])
    normalization_param.append(param)
print('normalization_param', np.array(normalization_param).shape, 
      np.isnan(np.array(normalization_param)).sum(), 
      np.isinf(np.array(normalization_param)).sum())

c = ConfigYAML()
c.normalization_param = normalization_param
c.save_file(DATADIR + 'config' + SAVE_NAME + '.yaml')

# plt.plot(np.arange(dynamics.shape[0]), dynamics[:, 0], '.', markersize=1)
# plt.show()
# plt.plot(np.arange(dynamics.shape[0]), dynamics[:, 1], '.', markersize=1)
# plt.show()
# plt.plot(np.arange(dynamics.shape[0]), dynamics[:, 2], '.', markersize=1)
# plt.show()

# c = ConfigJSON()
# c.load_file(TRAIN_DATADIR + '/config.json')

train_states_fric = []
train_controls_fric = []
train_dynamics_fric = []
train_labels_fric = []
for ind, friction_ in enumerate(flist):
    states_fric = all_friction_states[ind]
    controls_fric = all_friction_control[ind]
    
    train_states = []
    train_controls = []
    train_dynamics = []
    train_labels = []
    
    for segment_ind in range(states_fric.shape[0]):
    # for segment_ind in range(1):
        states = states_fric[segment_ind]
        controls = controls_fric[segment_ind]
        
        states = np.vstack([states[i:i+TRAIN_SEGMENT][None, :] for i in range(1, len(states)-TRAIN_SEGMENT+1, TRAIN_SEGMENT)])
        controls = np.vstack([controls[i:i+TRAIN_SEGMENT][None, :] for i in range(1, len(controls)-TRAIN_SEGMENT+1, TRAIN_SEGMENT)])
        dynamics = (states[:, 1:, :] - states[:, :-1, :]) / TIME_INTERVAL
        print(np.sum(dynamics * 0.1 + states[:, 0, :] - states[:, 1, :]))
        label = [ind] * dynamics.shape[0]
        
        # for ind2 in range(4):
        #     states[:, :, ind2] = dp.runtime_normalize(states[:, :, ind2], normalization_param[ind2])
        
        # for ind2 in range(4, 7):
        #     # print(dynamics[0, 0, ind2-4])
        #     dynamics[:, :, ind2-4] = dp.runtime_normalize(dynamics[:, :, ind2-4], normalization_param[ind2])
        # for ind2 in range(7, 9):
        #     controls[:, :, ind2-7] = dp.runtime_normalize(controls[:, :, ind2-7], normalization_param[ind2])
        
        
        # print('dynamics', dynamics.shape)
        # print('states', states.shape)
        
        train_states.append(states)
        train_controls.append(controls[..., 0:1, :])
        train_dynamics.append(dynamics)
        train_labels.append(label)
            
    train_states_fric.append(np.vstack(train_states))
    train_controls_fric.append(np.vstack(train_controls))
    train_dynamics_fric.append(np.vstack(train_dynamics))
    train_labels_fric.append(np.hstack(train_labels))
    
train_states_fric = np.array(train_states_fric)
train_controls_fric = np.array(train_controls_fric)
train_dynamics_fric = np.array(train_dynamics_fric)
train_labels_fric = np.array(train_labels_fric)
    
print('train_states', train_states_fric.shape)
print('train_controls_fric', train_controls_fric.shape)
print('train_dynamics_fric', train_dynamics_fric.shape)
print('train_labels', train_labels_fric.shape)



np.savez(DATADIR + 'train_data' + SAVE_NAME, 
         train_states=train_states_fric, 
         train_controls=train_controls_fric, 
         train_dynamics=train_dynamics_fric,
         train_labels=train_labels_fric)
