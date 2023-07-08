import os
import tensorflow as tf
import numpy as np

from tf_models.activations import swish, sine_act_omega
from tf_models.initializers import siren_initializer
import tf_models.tf_util as tf_util

from tf_train_eval.dvmlp_train_gm_dataset import convert_dataset_to_dvmlp

#import Fourier layer,  https://github.com/titu1994/tf_fourier_features
# from tf_fourier_features import FourierFeatureProjection
from tf_models.fourier_layers import FourierFeatureProjection



class dense_hypernet_v1():
    def __init__(self, opt=None):

        #set items required to build model
        self.opt = opt
        if self.opt is not None:
            self.set_activation()
 
    def set_activation(self):
        self.activation = tf_util.get_activation(self.opt['activation'])

    def build_model(self):
        self.layer_names = []

        #input layer
        input1 = tf.keras.Input(shape=(self.opt['inputs_dim']), name='inputs')
        output = None

        # Dense Layers Construction
        print('Constructing Dense Layers')
        for iDense in range(self.opt['n_layers_hidden']+1):
            layername = 'dense_{:1.0f}'.format(iDense)
            print(layername)
            self.layer_names.append(layername)

            units       = self.opt['hidden_dim'][iDense]
            activation  = self.activation

            if iDense == 0:
                output = tf.keras.layers.Dense( units       = units,
                                                activation  = activation,
                                                name        = layername)(input1)
            else:
                if iDense == (self.opt['n_layers_hidden']):
                    units = self.opt['outputs_dim']
                    if self.opt['is_linear_output']:
                        activation = tf.keras.activations.linear
                    else:
                        activation = self.activation
                    
                output = tf.keras.layers.Dense( units       = units,
                                                activation  = activation,
                                                name        = layername)(output)

        self.model = tf.keras.Model(inputs=[input1], outputs=output)
        self.model.summary()


def dense_hypernet_v1_opt(  network_class       = None,
                            name                = None,
                            n_layers_hidden     = None,
                            hidden_dim          = None,
                            inputs_dim          = None,
                            outputs_dim         = None,
                            activation          = None,
                            is_linear_output    = None,
                            inputs              = None,
                            outputs             = None,):
    
    model_opt = locals()
    return model_opt

# wrapper for spatial net - defined same way as dense hypernet
class dense_spatial_net_v1(dense_hypernet_v1):
    def __init__(self, opt=None):
        super(dense_spatial_net_v1, self).__init__(opt=opt)

    # def build_model(self):
    #     self.layer_names = []

    #     #input layer
    #     input1 = tf.keras.Input(shape=(None, self.opt['inputs_dim']), name='inputs')
    #     output = None

    #     # Dense Layers Construction
    #     print('Constructing Dense Layers')
    #     for iDense in range(self.opt['n_layers_hidden']+1):
    #         layername = 'dense_{:1.0f}'.format(iDense)
    #         print(layername)
    #         self.layer_names.append(layername)

    #         units       = self.opt['hidden_dim'][iDense]
    #         activation  = self.activation

    #         if iDense == 0:
    #             output = tf.keras.layers.Dense( units       = units,
    #                                             activation  = activation,
    #                                             name        = layername)(input1)
    #         else:
    #             if iDense == (self.opt['n_layers_hidden']):
    #                 units = self.opt['outputs_dim']
    #                 if self.opt['is_linear_output']:
    #                     activation = tf.keras.activations.linear
    #                 else:
    #                     activation = self.activation
                    
    #             output = tf.keras.layers.Dense( units       = units,
    #                                             activation  = activation,
    #                                             name        = layername)(output)

    #     self.model = tf.keras.Model(inputs=[input1], outputs=output)
    #     self.model.summary()

def dense_v1_options(   network_class       = None,
                        name                = None,
                        n_layers_hidden     = None,
                        hidden_dim          = None,
                        inputs              = None,
                        inputs_dim          = None,
                        outputs             = None,
                        outputs_dim         = None,
                        activation          = None,
                        is_linear_output    = None,):
    '''Template for model options for dense_v1 networks'''

    model_opt = locals()
    return model_opt

class dense_v1():
    def __init__(self, prob_set):
        '''stand-alone dense network'''
        self.prob_set   = prob_set
        self.opt        = prob_set.opt
        self.model_opt  = prob_set.opt['model_opt']
        self.call_backs = None

        self.set_activation()

    def set_activation(self):
        self.activation = tf_util.get_activation(self.model_opt['activation'])

    def set_save_paths(self):
        self.fn_csv                 = os.path.join(self.prob_set.model_path, 'training.csv')

        self.fn_weights_val_best    = os.path.join(self.prob_set.model_path, 'weights.val_best.h5')
        self.fn_weights_train_best  = os.path.join(self.prob_set.model_path, 'weights.train_best.h5')
        self.fn_weights_end         = os.path.join(self.prob_set.model_path, 'weights.end.h5')

        self.fn_model_val_best      = os.path.join(self.prob_set.model_path, 'model.val_best.tf')
        self.fn_model_train_best    = os.path.join(self.prob_set.model_path, 'model.train_best.tf')
        self.fn_model_end           = os.path.join(self.prob_set.model_path, 'model.end.tf')

        self.fn_history             = os.path.join(self.prob_set.model_path, 'history.pickle')

    def start_csv_logger(self, is_continue_training = False):
        '''Start the csv_logger, optionally appending if continuing training, append to self.callbacks
        '''
        
        csv_logger = tf.keras.callbacks.CSVLogger(self.fn_csv, append=is_continue_training)
        if self.call_backs is None:
            self.call_backs = []
        self.call_backs.append(csv_logger)
          
    def make_callbacks_weights(self):
        '''Make checkpoints to save the weights during training'''

        if self.call_backs is None:
            self.call_backs = []
        #best validation-loss weights
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.fn_weights_val_best, 
                                                        monitor            = 'val_loss',
                                                        verbose            = 1,
                                                        save_best_only     = True,
                                                        mode               = 'min',
                                                        save_weights_only  = True)
        self.call_backs.append(checkpoint)

        #save end weights - every epoch
        checkpoint_2 = tf.keras.callbacks.ModelCheckpoint(  self.fn_weights_end,
                                                            monitor             = 'val_loss',
                                                            verbose             = 1, 
                                                            save_best_only      = False,
                                                            save_freq           = 'epoch',
                                                            save_weights_only   = True)
        self.call_backs.append(checkpoint_2)

        #best training-loss weights
        checkpoint_3 = tf.keras.callbacks.ModelCheckpoint(  self.fn_weights_train_best, 
                                                            monitor             = 'loss',
                                                            verbose             = 1,
                                                            save_best_only      = True,
                                                            mode                = 'min',
                                                            save_weights_only   = True)
        self.call_backs.append(checkpoint_3)

    def make_callbacks_model(self):
        '''Make callbacks to save the model to file'''
        
        if self.call_backs is None:
            self.call_backs = []

        #best validation-loss model
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.fn_model_val_best, 
                                                        monitor             = 'val_loss',
                                                        verbose             = 1,
                                                        save_best_only      = True,
                                                        mode                = 'min',
                                                        save_weights_only   = False,
                                                        save_format         = '.tf')
        self.call_backs.append(checkpoint)

        #save model every epoch
        checkpoint_2 = tf.keras.callbacks.ModelCheckpoint(  self.fn_model_end,
                                                            monitor             = 'val_loss',
                                                            verbose             = 1,
                                                            save_best_only      = False,
                                                            save_weights_only   = False,
                                                            save_format         = '.tf')
        self.call_backs.append(checkpoint_2)

        #best training-loss model
        checkpoint_3 = tf.keras.callbacks.ModelCheckpoint(  self.fn_model_train_best, 
                                                            monitor             = 'loss',
                                                            verbose             = 1,
                                                            save_best_only      = True,
                                                            mode                = 'min',
                                                            save_weights_only   = False,
                                                            save_format         = '.tf')
        self.call_backs.append(checkpoint_3)

    def tensorboard_callbacks(self, histogram_freq=10, profile_batch=(1,5)):

        self.log_dir = os.path.join(self.prob_set.model_path, 'tb_logs', 'training')
        tb_callback  = tf.keras.callbacks.TensorBoard(  log_dir         = self.log_dir,
                                                        histogram_freq  = histogram_freq,
                                                        write_graph     = True,
                                                        profile_batch   = profile_batch )
        if self.call_backs is None:
            self.call_backs = []
        self.call_backs.append(tb_callback)

    def build_model(self,):
        self.layer_names = []

        #input layer
        input1 = tf.keras.Input(shape=(self.model_opt['inputs_dim']), name='inputs')
        output = None

        # Dense Layers Construction
        print('Constructing Dense Layers')
        for iDense in range(self.model_opt['n_layers_hidden']+1):
            layername = 'dense_{:1.0f}'.format(iDense)
            print(layername)
            self.layer_names.append(layername)

            units       = self.model_opt['hidden_dim'][iDense]
            activation  = self.activation

            if iDense == 0:
                output = tf.keras.layers.Dense( units       = units,
                                                activation  = activation,
                                                name        = layername)(input1)
            else:
                if iDense == (self.model_opt['n_layers_hidden']):
                    units = self.model_opt['outputs_dim']
                    if self.model_opt['is_linear_output']:
                        activation = tf.keras.activations.linear
                    else:
                        activation = self.activation
                    
                output = tf.keras.layers.Dense( units       = units,
                                                activation  = activation,
                                                name        = layername)(output)

        self.model = tf.keras.Model(inputs=[input1], outputs=output)
        self.model.summary()
############################################################################################################


def dense_v2_options(   network_class       = None,
                        name                = None,
                        n_layers_hidden     = None,
                        hidden_dim          = None,
                        inputs              = None,
                        inputs_dim          = None,
                        outputs             = None,
                        outputs_dim         = None,
                        activation          = None,
                        is_linear_output    = None,
                        is_layer_norm       = None,):
    '''Template for model options for dense_v1 networks'''

    model_opt = locals()
    return model_opt


class dense_v2(dense_v1):
    '''Layer normalization option to model '''
    def __init__(self, prob_set):
        super().__init__(prob_set)

    def build_model(self,):
        self.layer_names = []

        #input layer
        input1 = tf.keras.Input(shape=(self.model_opt['inputs_dim']), name='inputs')
        output = None

        # Hidden layers
        print('Constructing Dense Layers')
        for iDense in range(self.model_opt['n_layers_hidden']):
            layername = 'dense_{:1.0f}'.format(iDense)
            print(layername)
            self.layer_names.append(layername)

            units       = self.model_opt['hidden_dim'][iDense]
            activation  = self.activation

            if iDense == 0:
                output = tf.keras.layers.Dense( units       = units,
                                                activation  = activation,
                                                name        = layername)(input1)
            else:   
                output = tf.keras.layers.Dense( units       = units,
                                                activation  = activation,
                                                name        = layername)(output)

            # normalization layer
            if self.model_opt['is_layer_norm']:
                layername = 'layernorm_{:1.0f}'.format(iDense)
                print(layername)
                self.layer_names.append(layername)
                output = tf.keras.layers.LayerNormalization()(output)

        # output layer
        units = self.model_opt['outputs_dim']
        if self.model_opt['is_linear_output']:
            activation = tf.keras.activations.linear
        else:
            activation = self.activation
        layername = 'dense_{:1.0f}'.format(iDense+1)
        print(layername)

        output = tf.keras.layers.Dense( units       = units,
                                        activation  = activation,
                                        name        = layername)(output)
        
        
        self.model = tf.keras.Model(inputs=[input1], outputs=output)
        self.model.summary()

def siren_v1_options(   network_class       = None,
                        name                = None,
                        n_layers_hidden     = None,
                        hidden_dim          = None,
                        inputs              = None,
                        inputs_dim          = None,
                        outputs             = None,
                        outputs_dim         = None,
                        activation          = None,
                        is_linear_output    = None,
                        weight_init_scale   = None,
                        omega_0             = None):
    '''Template for model options for dense_v1 networks'''

    model_opt = locals()
    return model_opt


class siren_v1(dense_v1):
    def __init__(self, prob_set):
        super().__init__(prob_set = prob_set)

    def set_activation(self):
        self.activation = sine_act_omega(omega_0=self.opt['model_opt']['omega_0'])

    def build_model(self,):
        self.layer_names = []

        #input layer
        input1 = tf.keras.Input(shape=(self.model_opt['inputs_dim']), name='inputs')
        output = None

        # Dense Layers Construction
        print('Constructing Dense Layers')
        for iDense in range(self.model_opt['n_layers_hidden']+1):
            layername = 'dense_{:1.0f}'.format(iDense)
            print(layername)
            self.layer_names.append(layername)

            units       = self.model_opt['hidden_dim'][iDense]
            if iDense == 0:
                kernel_initializer = siren_initializer( fan_in      = self.model_opt['inputs_dim'],
                                                        scale       = self.model_opt['weight_init_scale'],
                                                        omega_0     = self.model_opt['omega_0'],
                                                        is_first    = True )
                output = tf.keras.layers.Dense( units               = units,
                                                activation          = 'linear',
                                                name                = layername,
                                                kernel_initializer  = kernel_initializer,
                                                bias_initializer    = kernel_initializer,)(input1)
                output = sine_act_omega(omega_0=self.opt['model_opt']['omega_0'])(output)
            
            # output layer
            # over-ride units, make sure it's correct per model settings
            # use glorot-uniform initializer if linear output layer
            elif iDense == (self.model_opt['n_layers_hidden']):
                units = self.model_opt['outputs_dim']
                if self.model_opt['is_linear_output']:
                    kernel_initializer = 'glorot_uniform'
                else:
                    kernel_initializer = siren_initializer( fan_in      = self.model_opt['hidden_dim'][iDense-1],
                                                            scale       = self.model_opt['weight_init_scale'],
                                                            omega_0     = self.model_opt['omega_0'],
                                                            is_first    = False )    
                output = tf.keras.layers.Dense( units               = units,
                                                activation          = 'linear',
                                                name                = layername,
                                                kernel_initializer  = kernel_initializer,
                                                bias_initializer    = kernel_initializer,)(output)
                if not self.model_opt['is_linear_output']:
                    output = sine_act_omega(omega_0=self.opt['model_opt']['omega_0'])(output)
            else:
                kernel_initializer = siren_initializer( fan_in      = self.model_opt['hidden_dim'][iDense-1],
                                                        scale       = self.model_opt['weight_init_scale'],
                                                        omega_0     = self.model_opt['omega_0'],
                                                        is_first    = False )
                output = tf.keras.layers.Dense( units               = units,
                                                activation          = 'linear',
                                                name                = layername,
                                                kernel_initializer  = kernel_initializer,
                                                bias_initializer    = kernel_initializer,)(output)
                output = sine_act_omega(omega_0=self.opt['model_opt']['omega_0'])(output)       

        self.model = tf.keras.Model(inputs=[input1], outputs=output)
        self.model.summary()

class denseV1Tuner():
    def __init__(self, prob_set, *args, **kwargs):
        '''stand-alone dense network'''
        self.prob_set   = prob_set
        self.opt        = prob_set.opt
        self.model_opt  = prob_set.opt['model_opt']
        self.call_backs = None
        self.set_activation()

    def set_activation(self):
        self.activation = tf_util.get_activation(self.model_opt['activation'])

    def build_model(self,):
        self.layer_names = []

        #input layer
        input1 = tf.keras.Input(shape=(self.model_opt['inputs_dim']), name='inputs')
        output = None

        # Dense Layers Construction
        print('Constructing Dense Layers')
        for iDense in range(self.model_opt['n_layers_hidden']+1):
            layername = 'dense_{:1.0f}'.format(iDense)
            print(layername)
            self.layer_names.append(layername)

            units       = self.model_opt['hidden_dim'][iDense]
            activation  = self.activation

            if iDense == 0:
                output = tf.keras.layers.Dense( units       = units,
                                                activation  = activation,
                                                name        = layername)(input1)
            else:
                if iDense == (self.model_opt['n_layers_hidden']):
                    units = self.model_opt['outputs_dim']
                    if self.model_opt['is_linear_output']:
                        activation = tf.keras.activations.linear
                    else:
                        activation = self.activation
                    
                output = tf.keras.layers.Dense( units       = units,
                                                activation  = activation,
                                                name        = layername)(output)

        self.model = tf.keras.Model(inputs=[input1], outputs=output)
        self.model.summary()


def fourierDense_v1_options(    network_class       = None,
                                name                = None,
                                n_layers_hidden     = None,
                                hidden_dim          = None,
                                inputs              = None,
                                inputs_dim          = None,
                                outputs             = None,
                                outputs_dim         = None,
                                activation          = None,
                                is_linear_output    = None,
                                n_fourier_features  = None,
                                fourier_gaussian_scale = None):
    '''Template for model options for dense_v1 networks'''

    model_opt = locals()
    return model_opt

class fourierDense_v1(tf.keras.Model):
    def __init__(self, prob_set, **kwargs):
        super().__init__(**kwargs)
        self.prob_set = prob_set
        
        #aliases to read 
        self.opt        = prob_set.opt
        self.model_opt  = prob_set.opt['model_opt']

        #set activation, build the model
        self.activation = tf_util.get_activation(self.model_opt['activation'])
        self.call_backs = None
        # don't build here to be compatible with existing code, without changes, dvmlp_train_gm_dataset.py
        # self.build_model()


    def call(self,x):
        y_pred = self.model(x)
        return(y_pred)

    def build_model(self,):
        self.layer_names = []
        #input and Fourier layers
        input1 = tf.keras.Input(shape=(self.model_opt['inputs_dim']), name='inputs')
        
        output = FourierFeatureProjection(  gaussian_projection = self.model_opt['n_fourier_features'], 
                                            gaussian_scale      = self.model_opt['fourier_gaussian_scale'],
                                            name                = 'fourier_0',)(input1)
        
        # Dense Layers Construction 
        # all consume and produce output, with optional linear output layer
        print('Constructing Dense Layers')
        for iDense in range(self.model_opt['n_layers_hidden']):
            layername = 'dense_{:1.0f}'.format(iDense)
            print(layername)
            self.layer_names.append(layername)
               
            output = tf.keras.layers.Dense( units       = self.model_opt['hidden_dim'][iDense],
                                            activation  = self.activation,
                                            name        = layername)(output)
            
        #output layer
        units = self.model_opt['outputs_dim']
        if self.model_opt['is_linear_output']:
            activation = tf.keras.activations.linear
        else:
            activation = self.activation
        output = tf.keras.layers.Dense( units       = units,
                                        activation  = activation,
                                        name        = 'dense_{:1.0f}'.format(iDense+1))(output)
                
        self.model = tf.keras.Model(inputs=[input1], outputs=output)
        self.model.summary()
        self.built = True


    def set_save_paths(self):
        self.fn_csv                 = os.path.join(self.prob_set.model_path, 'training.csv')

        self.fn_weights_val_best    = os.path.join(self.prob_set.model_path, 'weights.val_best.h5')
        self.fn_weights_train_best  = os.path.join(self.prob_set.model_path, 'weights.train_best.h5')
        self.fn_weights_end         = os.path.join(self.prob_set.model_path, 'weights.end.h5')

        self.fn_model_val_best      = os.path.join(self.prob_set.model_path, 'model.val_best.tf')
        self.fn_model_train_best    = os.path.join(self.prob_set.model_path, 'model.train_best.tf')
        self.fn_model_end           = os.path.join(self.prob_set.model_path, 'model.end.tf')

        self.fn_history             = os.path.join(self.prob_set.model_path, 'history.pickle')

    def start_csv_logger(self, is_continue_training = False):
        '''Start the csv_logger, optionally appending if continuing training, append to self.callbacks
        '''
        
        csv_logger = tf.keras.callbacks.CSVLogger(self.fn_csv, append=is_continue_training)
        if self.call_backs is None:
            self.call_backs = []
        self.call_backs.append(csv_logger)
          
    def make_callbacks_weights(self):
        '''Make checkpoints to save the weights during training'''

        if self.call_backs is None:
            self.call_backs = []
        #best validation-loss weights
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.fn_weights_val_best, 
                                                        monitor            = 'val_loss',
                                                        verbose            = 1,
                                                        save_best_only     = True,
                                                        mode               = 'min',
                                                        save_weights_only  = True)
        self.call_backs.append(checkpoint)

        #save end weights - every epoch
        checkpoint_2 = tf.keras.callbacks.ModelCheckpoint(  self.fn_weights_end,
                                                            monitor             = 'val_loss',
                                                            verbose             = 1, 
                                                            save_best_only      = False,
                                                            save_freq           = 'epoch',
                                                            save_weights_only   = True)
        self.call_backs.append(checkpoint_2)

        #best training-loss weights
        checkpoint_3 = tf.keras.callbacks.ModelCheckpoint(  self.fn_weights_train_best, 
                                                            monitor             = 'loss',
                                                            verbose             = 1,
                                                            save_best_only      = True,
                                                            mode                = 'min',
                                                            save_weights_only   = True)
        self.call_backs.append(checkpoint_3)

    def tensorboard_callbacks(self, histogram_freq=10, profile_batch=(1,5)):

        self.log_dir = os.path.join(self.prob_set.model_path, 'tb_logs', 'training')
        tb_callback  = tf.keras.callbacks.TensorBoard(  log_dir         = self.log_dir,
                                                        histogram_freq  = histogram_freq,
                                                        write_graph     = True,
                                                        profile_batch   = profile_batch )
        if self.call_backs is None:
            self.call_backs = []
        self.call_backs.append(tb_callback)


def fourierDense_v2_options(    network_class       = None,
                                name                = None,
                                n_layers_hidden     = None,
                                hidden_dim          = None,
                                inputs              = None,
                                inputs_dim          = None,
                                outputs             = None,
                                outputs_dim         = None,
                                activation          = None,
                                is_linear_output    = None,
                                mu_dim              = None,
                                x_dim               = None,
                                n_fourier_features  = None,
                                fourier_gaussian_scale = None):
    '''Template for model options for dense_v1 networks'''

    model_opt = locals()
    return model_opt


class fourierDense_v2(tf.keras.Model):
    '''Apply the random fourier features only to x, not to mu'''
    def __init__(self, prob_set, **kwargs):
        super().__init__(**kwargs)
        self.prob_set = prob_set
        
        #aliases to read 
        self.opt        = prob_set.opt
        self.model_opt  = prob_set.opt['model_opt']

        #set activation, build the model
        self.activation = tf_util.get_activation(self.model_opt['activation'])
        self.call_backs = None

        # don't build here to be compatible with existing code, without changes, dvmlp_train_gm_dataset.py
        # self.build_model()


    def call(self,x):
        y_pred = self.model(x)
        return(y_pred)

    def build_model(self,):

        #input layers
        input_mu = tf.keras.Input(shape=(self.model_opt['mu_dim']), name='input_mu')
        input_x  = tf.keras.Input(shape=(self.model_opt['x_dim']), name='input_x')

        # random fourier projection for spatial inputs
        output = FourierFeatureProjection(  gaussian_projection = self.model_opt['n_fourier_features'], 
                                            gaussian_scale      = self.model_opt['fourier_gaussian_scale'],
                                            name                = 'fourier_0',)(input_x)

        # concatenate mu with fourier-projected inputs
        output = tf.keras.layers.Concatenate()([input_mu, output])

        # Dense Layers Construction
        print('Constructing Dense Layers')
        for iDense in range(self.model_opt['n_layers_hidden']):
            layername = 'dense_{:1.0f}'.format(iDense)
            print(layername)
               
            output = tf.keras.layers.Dense( units       = self.model_opt['hidden_dim'][iDense],
                                            activation  = self.activation,
                                            name        = layername)(output)
            
        #output layer
        units = self.model_opt['outputs_dim']
        if self.model_opt['is_linear_output']:
            activation = tf.keras.activations.linear
        else:
            activation = self.activation
        output = tf.keras.layers.Dense( units       = units,
                                        activation  = activation,
                                        name        = 'dense_{:1.0f}'.format(iDense+1))(output)
                
        self.model = tf.keras.Model(inputs=[input_mu, input_x], outputs=output)
        self.model.summary()
        self.built = True


    def set_save_paths(self):
        self.fn_csv                 = os.path.join(self.prob_set.model_path, 'training.csv')

        self.fn_weights_val_best    = os.path.join(self.prob_set.model_path, 'weights.val_best.h5')
        self.fn_weights_train_best  = os.path.join(self.prob_set.model_path, 'weights.train_best.h5')
        self.fn_weights_end         = os.path.join(self.prob_set.model_path, 'weights.end.h5')

        self.fn_model_val_best      = os.path.join(self.prob_set.model_path, 'model.val_best.tf')
        self.fn_model_train_best    = os.path.join(self.prob_set.model_path, 'model.train_best.tf')
        self.fn_model_end           = os.path.join(self.prob_set.model_path, 'model.end.tf')

        self.fn_history             = os.path.join(self.prob_set.model_path, 'history.pickle')

    def start_csv_logger(self, is_continue_training = False):
        '''Start the csv_logger, optionally appending if continuing training, append to self.callbacks
        '''
        
        csv_logger = tf.keras.callbacks.CSVLogger(self.fn_csv, append=is_continue_training)
        if self.call_backs is None:
            self.call_backs = []
        self.call_backs.append(csv_logger)
          
    def make_callbacks_weights(self):
        '''Make checkpoints to save the weights during training'''

        if self.call_backs is None:
            self.call_backs = []
        #best validation-loss weights
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.fn_weights_val_best, 
                                                        monitor            = 'val_loss',
                                                        verbose            = 1,
                                                        save_best_only     = True,
                                                        mode               = 'min',
                                                        save_weights_only  = True)
        self.call_backs.append(checkpoint)

        #save end weights - every epoch
        checkpoint_2 = tf.keras.callbacks.ModelCheckpoint(  self.fn_weights_end,
                                                            monitor             = 'val_loss',
                                                            verbose             = 1, 
                                                            save_best_only      = False,
                                                            save_freq           = 'epoch',
                                                            save_weights_only   = True)
        self.call_backs.append(checkpoint_2)

        #best training-loss weights
        checkpoint_3 = tf.keras.callbacks.ModelCheckpoint(  self.fn_weights_train_best, 
                                                            monitor             = 'loss',
                                                            verbose             = 1,
                                                            save_best_only      = True,
                                                            mode                = 'min',
                                                            save_weights_only   = True)
        self.call_backs.append(checkpoint_3)

    def tensorboard_callbacks(self, histogram_freq=10, profile_batch=(1,5)):

        self.log_dir = os.path.join(self.prob_set.model_path, 'tb_logs', 'training')
        tb_callback  = tf.keras.callbacks.TensorBoard(  log_dir         = self.log_dir,
                                                        histogram_freq  = histogram_freq,
                                                        write_graph     = True,
                                                        profile_batch   = profile_batch )
        if self.call_backs is None:
            self.call_backs = []
        self.call_backs.append(tb_callback)
        

def get_f_dense(f_target):
    '''Return the class handle for a stand-alone dense network'''

    f_allowed = {   'dense_v1'          : dense_v1,
                    'dense_v2'          : dense_v2,
                    'siren_v1'          : siren_v1,
                    'denseV1Tuner'      : denseV1Tuner,
                    'fourierDense_v1'   : fourierDense_v1,
                    'fourierDense_v2'   : fourierDense_v2,
                }
    assert f_target in f_allowed.keys()
    return f_allowed[f_target]


