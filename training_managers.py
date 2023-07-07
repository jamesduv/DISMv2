import os
import json
import numpy as np
import hp5y as h5

import tf_util
import problem_settings as settings



class modelTrainer():
    def __init__(self, opt=None):
        self.opt = opt

    def set_data_paths(self):
        self.fn_gen_settings    = os.path.join( 'poisson_publish_v1', 'generator_settings.h5' )
        self.fn_dataset         = os.path.join( 'poisson_publish_v1', 'dataset.h5' )
        self.fn_data_stats      = os.path.join( 'poisson_publish_v1', 'data_stats.json')

    def train_dataset(  self,
                        save_path_base          = None, 
                        is_train_efficient      = True,
                        is_overwrite            = True,
                        is_mixed_precision      = True,
                        backend_precision_value = 'float64',
                        model_read_dir          = None,
                        is_checkpoint_model     = False,
                        is_downsample_val_data  = True,
                        val_data_downsample_factor = 0.1,
                        is_verify_custom        = False,
                        is_eager_execution      = False,
                        is_truncated_data       = True,):
        
        if not is_eager_execution:
            disable_eager_execution()

        # backend_precision_value only taken is is_mixed_precision is False
        tf_util.set_tensorflow_precision_policy(    backend_precision_value = backend_precision_value,
                                                    is_mixed_precision      = is_mixed_precision)

        # if opt is None, then continue training, load model settings from file
        if self.opt is None:
            assert model_read_dir is not None
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



        
    def load_dataset_h5(self,):
        with h5.File(fn_dataset, 'w') as fRead:
            X_train = fRead['X_train'][()]
            X_val   = fRead['X_val'][()]
            M_train = fRead['M_train'][()]
            M_val   = fRead['M_val'][()]
            Y_train = fRead['Y_train'][()]
            Y_val   = fRead['Y_val'][()]

        train_data = (X_train, M_train, Y_train)
        val_data = (X_val, M_val, Y_val)

        return train_data, val_data
    

        


