import os
import json
import numpy as np

# NpEncoder taken from:
# https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
# used for writing JSON files with numpy arrays
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class settings_base():
    '''Container to aid in creating and interacting with trained networks'''
    def __init__(self, opt = None):
        self.opt            = opt
        self.model_path     = None
        self.fn_stats       = None
    
    def make_model_save_dir(self, save_path_base, is_overwrite=False):
        '''Make the directory for saving a model and its weights, located at
            save_path_base/self.opt['name']

        Overwrite existing data if is_overwrite is True. If is_overwrite is false
        and path exists, raise exception.

        Args:
            opt (dict)              : dictionary containing the options for building the model
            save_path_base (str)    : path to directory where new directory is to be created
            is_overwrite (bool) : option specifying operation mode.
        '''

        save_dir = os.path.join(save_path_base, self.opt['model_opt']['name'])

        #delete save directory if it exists - only digs 1 layer deep
        if is_overwrite:
            if os.path.exists(save_dir):
                filelist = os.listdir(save_dir)
                for f in filelist:
                    os.remove(os.path.join(save_dir, f))
                os.rmdir(save_dir)
        if os.path.exists(save_dir):
            raise Exception('Path already exists: {}'.format(save_dir))
        else:
            os.mkdir(save_dir)

        self.model_path = save_dir

    def save_settings_json(self, is_overwrite_json=False):
        '''Save network settings to single .json, located at:
        self.model_path/model_settings.json '''

        if self.model_path is None:
            raise Exception('No save directory specified, as self.model.path')

        if self.opt is None:
            raise Exception('No data to write, self.opt is None')

        fn_settings = os.path.join(self.model_path, 'model_settings.json' )

        if not is_overwrite_json and os.path.exists(fn_settings):
            raise Exception('{} already exists, overwrite disabled. Set '.format(fn_settings))
        else:
            json.dump(self.opt, open(fn_settings, 'w'), indent=4, cls=NpEncoder)

    def load_opt(self, fn_settings):
        '''Populate self.opt with contents of file fn_settings'''

        self.opt = json.load(open(fn_settings, 'r'))
        
        #convert idx lists back to ndarrays
        keys_to_ndarray = ['idx_cases', 'idx_train', 'idx_val']
        for ii, key in enumerate(keys_to_ndarray):
            if self.opt['data_opt'][key] is not None:
                self.opt['data_opt'][key] = np.array(self.opt['data_opt'][key])

    def generate_train_val_idx(self, is_set_seed = True, seed_val = 0):
        '''Generate the indices for training and validation groups
        
        Args:
            is_set_seed (bool)  : option to set seed for repeatability
            seed_val (int)      : seed value
        '''

        n_cases             = self.opt['data_opt']['n_cases']
        training_fraction   = self.opt['train_opt']['training_fraction']
        n_train             = int(np.floor(n_cases*training_fraction))
        n_val               = int(n_cases - n_train)
        idx_range           = np.arange(n_cases)
        idx_cases           = idx_range.copy()
        np.random.shuffle(idx_range)
        idx_val             = idx_range[:n_val].copy()
        idx_train           = idx_range[n_val:].copy()

        idx_train   = np.sort(idx_train)
        idx_val     = np.sort(idx_val)

        self.opt['data_opt']['idx_cases']   = idx_cases.tolist()
        self.opt['data_opt']['idx_train']   = idx_train.tolist()
        self.opt['data_opt']['idx_val']     = idx_val.tolist()

    def save_data_stats_separate(self):
        assert self.model_path is not None
        assert 'data_stats' in self.opt['data_opt'].keys()
        self.fn_stats = os.path.join( self.model_path, 'data_stats.json', )

        json.dump(  self.opt['data_opt']['data_stats'], 
                    open(self.fn_stats, 'w'), 
                    indent  = 4, 
                    cls     = NpEncoder)