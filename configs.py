import copy
from math import pow, sqrt, pi
import numpy as np

### Default common config
config = {}
# - Outputs set up
config['verbose'] = False
config['save_every'] = 1000
config['save_final'] = True
config['save_train_data'] = True
config['evaluate_every'] = 100
config['out_dir'] = 'code_outputs'

# - Experiment set up
config['train_dataset_size'] = -1
config['use_anomalous'] = False
seed = 1234
# np.random.seed(seed)
config['theta'] = np.random.uniform(0,pi)
np.random.seed()
config['model'] = 'affine' # affine, nonaffine
config['use_trained'] = False # train from pre-trained model

# - Opt set up
config['optimizer'] = 'adam' # adam, sgd
config['adam_beta1'] = 0.9
config['adam_beta2'] = 0.999
config['lr'] = 0.001
config['lr_decay'] = False
config['lr_adv'] = 1e-08

# - Obj set up
config['score_non_linear'] = 'linear' #linear, cubic, sinh
# if config['score_non_linear']=='linear':
#     config['clip_score'] = True
# else:
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

# - NN set up
config['init_bias'] = 0.0
config['w_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config['w_init_std'] = 0.0099999
config['d_init'] = 'normal' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config['d_init_std'] = 0.00099999
config['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config['conv_init'] = 'glorot_uniform' #he, glorot, normilized_glorot, truncated_norm
config['nonaffine_nlayers'] = 2
config['nonaffine_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config['nonaffine_init_std'] = 0.0099999
config['nonaffine_non_linear'] = 'tanh'
config['nonaffine_eta1'] = 1.
config['nonaffine_eta2'] = 1.

### Line config
config_line = config.copy()
config_line['dataset'] = 'line'

### Quadratic config
config_quadratic = config.copy()
config_quadratic['dataset'] = 'quadratic'
