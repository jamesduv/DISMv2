import os
import pickle
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

#import models, tools, dataloaders, etc
import problem_settings as settings
import hypernet_oneshot_networks as hypernet
import hypernet_oneshot_common as hypernet_common
import tf_util
import learning_rate_schedules
from dataloader_poisson import poissonDatasetLoader

def train_dataset(  opt                     = None,
                    save_path_base          = None, 
                    is_train_efficient      = True,
                    is_overwrite            = True,
                    is_mixed_precision      = True,
                    backend_precision_value = 'float64',
                    model_read_dir          = None,
                    is_checkpoint_model     = False,
                    is_downsample_val_data  = True,
                    val_data_downsample_factor = 0.1,
                    is_eager_execution      = False,):

    if not is_eager_execution:
        disable_eager_execution()

    # backend_precision_value only taken is is_mixed_precision is False
    tf_util.set_tensorflow_precision_policy(    backend_precision_value = backend_precision_value,
                                                is_mixed_precision      = is_mixed_precision)

     # if opt is None, then continue training, load model settings from file
    if opt is None:
        assert model_read_dir is not None
        fn_settings = os.path.join(model_read_dir, 'model_settings.json' )
        opt = json.load( open(fn_settings, 'r'))
        prob_set = settings.settings_base(opt = opt)
        prob_set.model_path = model_read_dir
        is_continue_training = True
    else:
        prob_set = settings.settings_base(opt = opt)
        prob_set.make_model_save_dir(save_path_base     = save_path_base,
                                        is_overwrite    = is_overwrite )

        prob_set.save_settings_json(is_overwrite_json = True)
        is_continue_training = False

    dataloader = poissonDatasetLoader()
    train_data, val_data = dataloader.load_dataset_h5()
    generator_settings = dataloader.load_generator_settings_json()
    data_stats = dataloader.load_data_stats_json()

    # fn_dataset      = os.path.join( os.environ['CFD_DATA_PATH'], 'poisson2d_datasets', opt['data_opt']['dataset'], 'dataset.pickle' )
    # fn_data_stats   = os.path.join( os.environ['CFD_DATA_PATH'], 'poisson2d_datasets', opt['data_opt']['dataset'], 'data_stats.pickle' )
    # fn_gen_settings = os.path.join( os.environ['CFD_DATA_PATH'], 'poisson2d_datasets', opt['data_opt']['dataset'], 'generator_settings.pickle' )
    # dataset = pickle.load( open(fn_dataset, 'rb'))
    # generator_settings = pickle.load(open(fn_gen_settings, 'rb'))
    
    # train_data  = dataset['train_data']
    # val_data    = dataset['val_data']


    if is_downsample_val_data:
        val_data = downsample_val_data_v2(  val_data                    = val_data, 
                                            val_data_downsample_factor  = val_data_downsample_factor,
                                            n_shape_classes             = generator_settings['n_shape_classes'])


    n_updates_per_epoch = int(np.ceil(train_data[0].shape[0] / prob_set.opt['train_opt']['batch_size'] ) )

    #update model settings to include data stats, also save separately
    prob_set.opt['data_opt']['data_stats'] = data_stats
    prob_set.save_settings_json(is_overwrite_json = True)
    prob_set.save_data_stats_separate()

    #get the overall hypernet model (net) and class container for keras hypernetwork
    net, hypernet_class = hypernet_common.get_dense_hypermodel(prob_set)
    
    #prepare for training
    net.set_save_paths()
    net.start_csv_logger( is_continue_training = is_continue_training)
    net.make_callbacks_weights()
    if prob_set.opt['train_opt']['is_profile']:
        net.tensorboard_callbacks( histogram_freq   = prob_set.opt['train_opt']['histogram_freq'], 
                                    profile_batch   = prob_set.opt['train_opt']['profile_batch'])

    if is_checkpoint_model:
        net.make_callbacks_model()

    # get the loss function
    f_loss = tf_util.get_loss(prob_set.opt['train_opt']['loss'])

    # get the optimizer
    f_optimizer = tf_util.get_optimizer_handle(prob_set.opt['train_opt']['optimizer'])

    # set learning rate or learning rate schedule
    if prob_set.opt['train_opt']['is_learning_rate_decay']:
        if prob_set.opt['train_opt']['learning_rate_decay_type'] == 'exponential_decay':
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay( 
                initial_learning_rate = prob_set.opt['train_opt']['learning_rate'],
                decay_steps = prob_set.opt['train_opt']['decay_steps'],
                decay_rate  = prob_set.opt['train_opt']['decay_rate'])
        elif prob_set.opt['train_opt']['learning_rate_decay_type'] == 'piecewise_const_exponential_decay':
            n_steps_const   = prob_set.opt['train_opt']['const_epochs'] * n_updates_per_epoch
            n_steps_decay   = prob_set.opt['train_opt']['decay_epochs'] * n_updates_per_epoch
            decay_rate      = prob_set.opt['train_opt']['decay_rate']
            initial_learning_rate = prob_set.opt['train_opt']['learning_rate']
            lr_schedule = learning_rate_schedules.piecewiseConstantExpDecay(   initial_learning_rate = initial_learning_rate,
                                                                               n_steps_const    = n_steps_const,
                                                                               n_steps_decay    = n_steps_decay,
                                                                               decay_rate       = decay_rate )
        else:
            lr_schedule = prob_set.opt['train_opt']['learning_rate']
    else:
        lr_schedule = prob_set.opt['train_opt']['learning_rate']

    optimizer = f_optimizer(learning_rate = lr_schedule)

    # if continuing training, load the previous end weights
    if is_continue_training:
        #must call model before loading weights-only from .h5
        xfake   = np.zeros((1,1,prob_set.opt['model_opt']['inputs_dim_main']))
        mufake  = np.zeros((1, prob_set.opt['model_opt']['inputs_dim_hypernet']))
        net([xfake, mufake])
        net.load_weights( net.fn_weights_end)

    # compile the model w/optimizer and loss, train the model
    net.compile(optimizer   = optimizer,
                loss        = f_loss)

    inputs_val = [val_data[0], val_data[1]]

    history = net.fit(  x   = [train_data[0], train_data[1]],
                        y   = train_data[2],
                        epochs      = net.opt['train_opt']['epochs'],
                        callbacks   = net.call_backs,
                        batch_size  = net.opt['train_opt']['batch_size'],
                        validation_data = (inputs_val, val_data[2]),
                        )

def downsample_val_data_v2(val_data, val_data_downsample_factor, n_shape_classes=8):
    '''Downsample the validation data by removing cases. Randomly remove the same number of cases from each shape class
    This assumes the validation data is ordered
    
    The data is truncated and comes stored as tuples of arrays
    '''
    n_cases_val = val_data[0].shape[0]
    n_cases_per_shape = n_cases_val / n_shape_classes

    n_cases_keep_shape = int(np.floor(n_cases_per_shape * val_data_downsample_factor))
    n_cases_keep_total = n_shape_classes * n_cases_keep_shape
    idx_shape = np.arange(n_cases_per_shape, dtype=int)
    np.random.shuffle(idx_shape)

    idx_keep_shape = idx_shape[:n_cases_keep_shape]
    idx_keep_shape.sort()
    idx_keep_all = []
    offset = int(0) 
    for iShape in np.arange(n_shape_classes):
        idx_cur = offset + idx_keep_shape
        idx_keep_all.append(idx_cur)
        offset += n_cases_per_shape

    idx_keep_all = np.concatenate(idx_keep_all, axis=0)
    idx_keep_all = idx_keep_all.astype('int')

    new_data = (val_data[0][idx_keep_all,...], val_data[1][idx_keep_all,...], val_data[2][idx_keep_all,...])

    return new_data

def  downsample_val_data(val_data, val_data_downsample_factor, n_shape_classes = 8):
    '''Downsample the validation data by removing cases. Randomly remove the same number of cases from each shape class
    This assumes the validation data is ordered
    
    The data is non-truncated and comes stored as a list of array tuples
    '''
    n_cases_val = len(val_data)
    n_cases_per_shape = n_cases_val / n_shape_classes

    n_cases_keep_shape = int(np.floor(n_cases_per_shape * val_data_downsample_factor))
    n_cases_keep_total = n_shape_classes * n_cases_keep_shape
    idx_shape = np.arange(n_cases_per_shape, dtype=int)
    np.random.shuffle(idx_shape)

    idx_keep_shape = idx_shape[:n_cases_keep_shape]
    idx_keep_shape.sort()
    idx_keep_all = []
    offset = int(0) 
    for iShape in np.arange(n_shape_classes):
        idx_cur = offset + idx_keep_shape
        idx_keep_all.append(idx_cur)
        offset += n_cases_per_shape

    idx_keep_all = np.concatenate(idx_keep_all, axis=0)
    idx_keep_all = idx_keep_all.astype('int')

    new_data = []
    for ii, idx in enumerate(idx_keep_all):
        new_data.append(val_data[idx])

    return new_data