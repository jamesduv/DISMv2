import os
import json
import numpy as np


import hypernet_oneshot_networks as hypernet_networks
import dense_networks

import hypernet_oneshot_train_poisson_dataset as train_model
import train_util

save_path_base = os.environ['HYPERNET_PATH']

model_opt = {   'network_class'         : 'keras_oneshot_v2',
                'name'                  : 'newRepo_DVH_test',
                'n_layers_hidden_main'  : 5,
                'hidden_dim_main'       : 50,
                'inputs_dim_main'       : 3,
                'outputs_dim_main'      : 1,
                'inputs_dim_hypernet'   : 12,
                'activation_main'       : 'swish',
                'is_linear_output_main' : True,
                'inputs_main'           : ['xc', 'yc', 'sdf_overall'],
                'inputs_hypernet'       : ['class_label_vec', 'xcen', 'ycen', 'radius', 'rotation',],
                'outputs_main'          : ['q',],
                'hypernet_opt'          : None,
                'weights_info_main'     : None,
            }
model_opt = hypernet_networks.v1_options_simple(**model_opt)

hypernet_opt = {'network_class'     : 'dense_hypernet_v1',
                'name'              : '{}_hypernetwork'.format(model_opt['name']),
                'n_layers_hidden'   :  5,
                'hidden_dim'        : [50, 50, 50, 50, 50, 1], #make len(hidden_dim) = n_layers_hidden+1, final entry is overwritten when building model
                'inputs_dim'        : model_opt['inputs_dim_hypernet'],
                'outputs_dim'       : model_opt['outputs_dim_hypernet'],
                'activation'        : 'swish',
                'is_linear_output'  : True,
                'inputs'            : model_opt['inputs_hypernet'],
                'outputs'           : ['main_net_weights',]
                }

hypernet_opt = dense_networks.dense_hypernet_v1_opt(**hypernet_opt)
model_opt['hypernet_opt'] = hypernet_opt

train_opt = {   'training_fraction'         : 0.2,
                'epochs'                    : 2,
                'batch_size'                : 4,
                'loss'                      : 'mse',
                'optimizer'                 : 'Adam',
                'learning_rate'             : 1e-04,
                'is_learning_rate_decay'    : True,
                'learning_rate_decay_type'  : 'piecewise_const_exponential_decay',
                'const_epochs'              : 500,
                'decay_epochs'              : 250,
                'decay_rate'                : 0.1,
                'kernel_regularizer'        : None,
                'x_norm_method'             : 'minmax',
                'mu_norm_method'            : 'minmax',
                'y_norm_method'             : 'minmax',
                'is_profile'                : False,
                'profile_batch'             : (1,100),
                'histogram_freq'            : 0,
            }

train_opt = train_util.train_opt_xmuy(**train_opt)

data_opt    =   {   'dataset'    : 'poisson_publish_v1',
                }

opt =   {   'model_opt'     : model_opt,
            'data_opt'      : data_opt,
            'train_opt'     : train_opt
        }


train_model.train_dataset(      opt                 = opt,
                                save_path_base      = save_path_base,
                                is_train_efficient  = True,
                                is_overwrite        = True,
                                model_read_dir      = None,
                                is_mixed_precision  = False,
                                is_checkpoint_model = False,
                                is_downsample_val_data = True,
                                val_data_downsample_factor = 0.06,
                                is_verify_custom    = False,
                                is_eager_execution  = False,
                                is_truncated_data   = True,)