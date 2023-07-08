import os
import json
import pickle
import h5py as h5

class poissonDatasetLoader():
    def __init__(self, dataset_path = None ):
        if dataset_path is None:
            self.fn_gen_settings_full   = os.path.join( 'poisson_publish_v1', 'generator_settings.pickle' )
            self.fn_dataset             = os.path.join( 'poisson_publish_v1', 'dataset.h5' )
            self.fn_data_stats          = os.path.join( 'poisson_publish_v1', 'data_stats.pickle')
        else:
            self.fn_gen_settings_full    = os.path.join(dataset_path, 'poisson_publish_v1', 'generator_settings.pickle' )
            self.fn_dataset         = os.path.join(dataset_path, 'poisson_publish_v1', 'dataset.h5' )
            self.fn_data_stats      = os.path.join(dataset_path, 'poisson_publish_v1', 'data_stats.pickle')

        self.fn_gen_settings_json   = self.fn_gen_settings_full.replace('pickle', 'json')
        self.fn_data_stats_json = self.fn_data_stats.replace('pickle', 'json')


    def load_dataset_h5(self,):
        with h5.File(self.fn_dataset, 'r') as fRead:
            X_train = fRead['X_train'][()]
            X_val   = fRead['X_val'][()]
            M_train = fRead['M_train'][()]
            M_val   = fRead['M_val'][()]
            Y_train = fRead['Y_train'][()]
            Y_val   = fRead['Y_val'][()]

        train_data = (X_train, M_train, Y_train)
        val_data = (X_val, M_val, Y_val)

        return train_data, val_data

    def load_generator_settings_json(self,):
        '''Load the generator settings without the truncated indices, much faster'''
        gen_settings = json.load(open(self.fn_gen_settings_json, 'r'))
        return gen_settings

    def load_generator_settings_full(self,):
        '''Load the generator settings including truncated indices, slower'''
        gen_settings = pickle.load(open(self.fn_gen_settings_full, 'rb'))
        return gen_settings

    def load_data_stats_json(self,):
        data_stats = json.load( open(self.fn_data_stats_json, 'r') )
        return data_stats

    def load_data_stats_pickle(self,):
        data_stats = pickle.load(open(self.fn_data_stats, 'rb'))
        return data_stats

    