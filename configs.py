import copy
from math import pow, sqrt, pi
import numpy as np

# Helper to define polynomial params for data
def init_poly(n):
    coef = [np.float32(np.random.uniform(low=5, high=11, size=None))*(2*np.random.randint(0, 2)-1) for _ in range(n+1)]
    return coef

### Default common config
config = {}
# - Outputs set up
config['verbose'] = False
config['save_every'] = 1000
config['save_final'] = True
config['save_train_data'] = True
config['evaluate_every'] = 100
config['out_dir'] = 'code_outputs'
config['hm_lim'] = 10

# - Experiment set up
config['train_dataset_size'] = -1
config['use_anomalous'] = False
config['fixed_dataset'] = False
seed = 1234
config['flow'] = 'planar' # identy, mlp, planar, realNVP, glow
config['use_trained'] = False # train from pre-trained model

# - Opt set up
config['optimizer'] = 'adam' # adam, sgd
config['adam_beta1'] = 0.9
config['adam_beta2'] = 0.999
config['lr'] = 0.001
config['lr_decay'] = False
config['lr_adv'] = 1e-08

# - Obj set up
config['score_nonlinear'] = 'linear' #linear, cubic, sinh
# if config['score_non_linear']=='linear':
#     config['clip_score'] = True
# else:
config['rotate'] = True
config['clip_score'] = False
config['clip_score_value'] = 1000.

# - D set up
config['train_d'] = True #learning D
if config['train_d']:
    config['lmbda'] = 1.
else:
    config['lmbda'] = 0.
config['d_const'] = False
config['d_reg'] = 'alpha' # trace, frob, det, alpha
config['d_reg_value'] =  10.0 #only used to reg. alpha
config['clip_d_reg'] = False
config['clip_d_reg_value'] = 10.

# - W set up
config['train_w'] = True #learning final w
if config['train_w']:
    config['gamma'] = 1.
else:
    config['gamma'] = 0.
config['w_reg'] = 'l2sq' # l1, l2, l2sq
config['clip_w_reg'] = True
config['clip_w_reg_value'] = 100.

# - params set up
config['w_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config['w_init_std'] = 0.0099999
config['d_init'] = 'normal' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config['d_init_std'] = 0.0099999

# - flows set up
config['normalization'] = 'batchnorm'
config['batch_norm_eps'] = 1e-05
config['batch_norm_momentum'] = 0.99
config['permutation'] = 'reverse'

# - NN set up
config['mlp_nlayers'] = 3
config['mlp_nunits'] = 128
config['mlp_init'] = 'glorot_uniform' #constant, normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config['mlp_init_std'] = 0.0099999
config['mlp_init_bias'] = 0.0
config['mlp_nonlinear'] = 'relu'
config['mlp_init_final'] = 'constant' #constant, normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config['mlp_init_std_final'] = 0.0099999
config['mlp_init_bias_final'] = 0.0
config['mlp_nonlinear_final'] = 'linear'
config['mlp_eta1'] = 1.
config['mlp_eta2'] = 1.

### Line config
config_line = config.copy()
config_line['dataset'] = 'line'
# sampling the fix params of the linear nominal
config_line['coef'] = init_poly(0)
config_line['theta'] = np.random.uniform(0,pi)


### Quadratic config
config_quadratic = config.copy()
config_quadratic['dataset'] = 'quadratic'
# sampling the fix params of the quadratic nominal
config_quadratic['coef'] = init_poly(2)
config_quadratic['theta'] = np.random.uniform(high=2*pi, size=None)

### Cubic config
config_cubic = config.copy()
config_cubic['dataset'] = 'cubic'
# sampling the fix params of the cubic nominal
config_cubic['coef'] = init_poly(3)
config_cubic['theta'] = np.float32(np.random.uniform(high=2*pi, size=None))

### Roll config
config_roll = config.copy()
config_roll['dataset'] = 'roll'
config_roll['theta'] = pi / 2.
