import os
import json
import numpy as np
import hp5y as h5
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution


import tf_util
import problem_settings as settings


class modelTrainerPoisson():
    def __init__(self, opt                      = None,
                        save_path_base          = None,
                        is_hypernet             = True, 
                        is_train_efficient      = True,
                        is_overwrite            = False,
                        is_mixed_precision      = True,
                        backend_precision_value = 'float64',
                        model_read_dir          = None,
                        is_checkpoint_model     = False,
                        is_downsample_val_data  = True,
                        val_data_downsample_factor = 0.1,
                        is_eager_execution      = False,):
        all_args = locals()
        all_args.pop('self')
        self.opt = opt
        self.allOpt = all_args

        self.save_path_base             = save_path_base
        self.is_overwrite               = is_overwrite
        self.is_mixed_precision         = is_mixed_precision
        self.backend_precision_value    = backend_precision_value
        self.model_read_dir             = model_read_dir
        self.is_checkpoint_model        = is_checkpoint_model
        self.is_downsample_val_data     = is_downsample_val_data
        self.val_data_downsample_factor = val_data_downsample_factor
        self.is_eager_execution         = is_eager_execution


    def set_data_paths(self):
        self.fn_gen_settings    = os.path.join( 'poisson_publish_v1', 'generator_settings.h5' )
        self.fn_dataset         = os.path.join( 'poisson_publish_v1', 'dataset.h5' )
        self.fn_data_stats      = os.path.join( 'poisson_publish_v1', 'data_stats.json')

    def train_dataset( self,):
        
        if not self.is_eager_execution:
            disable_eager_execution()

        # backend_precision_value only taken is is_mixed_precision is False
        tf_util.set_tensorflow_precision_policy(    backend_precision_value = self.backend_precision_value,
                                                    is_mixed_precision      = self.is_mixed_precision)

        # if opt is None, then continue training, load model settings from file
        if self.opt is None:
            assert self.model_read_dir is not None
            fn_settings = os.path.join(model_read_dir, 'model_settings.json' )
            self.opt = json.load( open(fn_settings, 'r'))
            prob_set = settings.settings_base(opt = opt)
            prob_set.model_path = model_read_dir
            is_continue_training = True
        else:
            prob_set = settings.settings_base(opt = self.opt)
            prob_set.make_model_save_dir(save_path_base     = save_path_base,
                                            is_overwrite    = is_overwrite )

            prob_set.save_settings_json(is_overwrite_json = True)
            is_continue_training = False

        self.set_data_paths()
        train_data, val_data = self.load_dataset_h5()

        if self.is_downsample_val_data:
            val_data = self.downsample_val_data(    val_data                    = val_data, 
                                                    val_data_downsample_factor  = val_data_downsample_factor,
                                                    n_shape_classes             = 8,)

        n_updates_per_epoch = int(np.ceil(train_data[0].shape[0] / prob_set.opt['train_opt']['batch_size'] ) )

        #update model settings to include data stats, also save separately
        prob_set.opt['data_opt']['data_stats'] = data_stats
        prob_set.save_settings_json(is_overwrite_json = True)
        prob_set.save_data_stats_separate()


        
    def load_dataset_h5(self,):
        with h5.File(self.fn_dataset, 'w') as fRead:
            X_train = fRead['X_train'][()]
            X_val   = fRead['X_val'][()]
            M_train = fRead['M_train'][()]
            M_val   = fRead['M_val'][()]
            Y_train = fRead['Y_train'][()]
            Y_val   = fRead['Y_val'][()]

        train_data = (X_train, M_train, Y_train)
        val_data = (X_val, M_val, Y_val)

        return train_data, val_data


    def downsample_val_data(self, val_data, val_data_downsample_factor, n_shape_classes=8):
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
    

        


