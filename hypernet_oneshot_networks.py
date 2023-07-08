import os
import tensorflow as tf
import numpy as np

import tf_util
from fourier_layers import FourierFeatureProjection

class keras_oneshot_v1(tf.keras.Model):
    '''Simple implementation of keras hypernetwork, without using
    tf.keras layers/models for the main network and subclassing tf.keras.Model

    Train with fully mixed batches

    Attributes:
        self.hypernet (tf.keras model)  : tf.keras hypernetwork, outputs tensor wts = ncases x nwt
        self.opt (dict)                 : all model options
        self.split_sizes (list of int)  : weight dimensions, used to split tensor wts
        self.wt_shapes (list of tuple)  : shape of weight element as tuples, with leading -1, as in (-1, dim1, dim2) to account for batch axis. Axes with -1 are dynamically allocated as required
        '''
        
    def __init__(self, prob_set, hypernet, **kwargs):
        super().__init__(**kwargs)
        self.prob_set       = prob_set
        self.opt            = prob_set.opt
        self.model_opt      = prob_set.opt['model_opt']
        self.split_sizes    = prob_set.opt['model_opt']['weights_info_main']['weights_split_sizes']
        self.weights_shapes = prob_set.opt['model_opt']['weights_info_main']['weights_shapes']
        self.hypernet   = hypernet
        self.call_backs = None

        self.set_activations()


    def call(self,x):
        return(self.call_model(x = x[0], mu = x[1]))
    
    def call_model(self, x, mu, training=False):
        #call hypernet
        weights = self.hypernet(mu, training=training)

        #split output, reshape into weight/bias matrices/vectors
        split_weights = tf.split(weights, self.split_sizes, axis=1)
        weights_reshaped = []
        for iShape, target_shape in enumerate(self.weights_shapes):
            cur_weight = tf.reshape(split_weights[iShape], target_shape)
            weights_reshaped.append(cur_weight)

        #forward propagate the main network
        output  = []
        output.append(x)
        for iLayer in np.arange(self.model_opt['n_layers_hidden_main']+1):
            
            #count by 2's to extract weight, bias from wt_shp
            idx_start   = 2*iLayer
            Wcur        = weights_reshaped[idx_start]
            bcur        = weights_reshaped[idx_start + 1]

            #layer multiplication
            zh = tf.einsum('ijk,ik->ij', Wcur, output[iLayer]) + bcur
            hcur = self.activations[iLayer](zh)
            output.append(hcur)

        return output[-1]

    def set_activations(self):
        '''Generate list of main network activation functions. All hidden layers
        use same activation function, output layer may be linear
        
        List is stored as self.activations
        '''
   
        f_hidden    = tf_util.get_activation(self.model_opt['activation_main'])
        activ       = [f_hidden,]*self.model_opt['n_layers_hidden_main']

        if self.model_opt['is_linear_output_main']:
            activ.append( tf.keras.activations.linear)
        else:
            activ.append(f_hidden)
        self.activations = activ

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

    def compose_main_network(self, mu):
        '''Given a design variable vector mu, generate the main network weights using
        the hypernetwork, load into Keras model and return
        
        Args:
            mu (tensor or ndarray)  : design variable vector

        Returns:
            model (tf.keras.Model)  : composed main network, w/generated weights
        '''

        weights         = self.hypernet(mu)
        n_layers_total  = self.model_opt['n_layers_hidden_main'] + 1

        #split output, reshape into weight/bias matrices/vectors
        split_weights = tf.split(weights, self.split_sizes, axis=1)
        wt_shp = []
        wt_squeeze = []
        for iShape, target_shape in enumerate(self.weights_shapes):
            cur_weight = tf.reshape(split_weights[iShape], target_shape)
            wt_shp.append(cur_weight)

            wt_sq = np.squeeze(cur_weight, axis=0)
            wt_squeeze.append(wt_sq.T)

        #make list of lists, gathering weights/biases by layer
        #Example: wts_all[0] = (W0, b0)
        wts_all = []
        for iLayer in np.arange(n_layers_total):
            idx1 = iLayer*2
            idx2 = idx1 + 1    
            curlayer_wt = [wt_squeeze[idx1], wt_squeeze[idx2]]
            wts_all.append(curlayer_wt)

        #build the main network as a tf.keras.Model
        #input layer
        input1 = tf.keras.Input(shape=(self.model_opt['inputs_dim_main']), name='input')
        output = None

        activ = tf_util.get_activation(self.model_opt['activation_main'])
        units = self.model_opt['hidden_dim_main']

        # Dense Layers Construction
        print('Constructing Dense Layers')
        for iLayer in range(n_layers_total):
            layername = 'dense_{:1.0f}'.format(iLayer)
            print(layername)

            activ = self.activations[iLayer]
            if iLayer == (n_layers_total -1):
                units = self.model_opt['outputs_dim_main']
            
            if iLayer == 0:
                output = tf.keras.layers.Dense( units       = units,
                                                activation  = activ,
                                                name        = layername)(input1)
            else:   
                output = tf.keras.layers.Dense(units         = units,
                                                activation   = activ,
                                                name         = layername)(output)
        
        model = tf.keras.Model(inputs=[input1], outputs=output)
        model.summary()

        model_weights = model.get_weights()

        #load the weights into the model
        for iLayer, curwts in enumerate(wts_all):
            curlayer = model.get_layer(index = iLayer+1) #iLayer+1 to skip input layer
            curlayer.set_weights(curwts)

        return model


def v1_options_simple(  network_class           = None,
                        name                    = None,
                        n_layers_hidden_main    = None,
                        hidden_dim_main         = None,
                        inputs_dim_main         = None,
                        outputs_dim_main        = None,
                        inputs_dim_hypernet     = None,
                        outputs_dim_hypernet    = None,
                        activation_main         = None,
                        is_linear_output_main   = None,
                        inputs_main             = None,
                        inputs_hypernet         = None,
                        outputs_main            = None,
                        hypernet_opt            = None,
                        weights_info_main       = None,
                        ):
    '''Template for model options for keras_oneshot_v1
    Compute the main network weight shapes if not supplied, assuming constant 
    hidden dimension through the main network
    '''

    if weights_info_main is None:
        weights_info_main = v1_weight_dimensions_simple(n_layers_hidden = n_layers_hidden_main,
                                                        hidden_dim      = hidden_dim_main,
                                                        inputs_dim      = inputs_dim_main,
                                                        outputs_dim     = outputs_dim_main)
        outputs_dim_hypernet = weights_info_main['weights_total_dim']
    model_opt = locals()
    return model_opt

def v1_weight_dimensions_simple(n_layers_hidden = 3,
                                hidden_dim      = 10,
                                inputs_dim      = 1,
                                outputs_dim     = 1):
    '''
    Given:
        -the options for the main dense network, simply defined with the 
        same number of hidden nodes in all layers, 
    Compute:
        - the total number of weights/biases required from the hypernet,
        - the sizes and shapes of each weight matrix and bias vector
    Args:
        n_layers_hidden (int)  
        hidden_dim      (int)
        inputs_dim      (int)
        outputs_dim     (int)
    Returns:
        weights_info    (dict)
    '''

    #total number of weights to generate
    wt_input_dim  = hidden_dim*inputs_dim + hidden_dim      #first hidden layer # weights
    wt_hidden_dim = hidden_dim**2 + hidden_dim              #other hidden layer # weights
    wt_output_dim = hidden_dim*outputs_dim + outputs_dim    #output layer # weights
    wt_total_dim  = wt_input_dim + ((n_layers_hidden-1) * wt_hidden_dim) + wt_output_dim
    
    #store shapes of each weight matrix and bias vector
    #leading -1 accounts for batch axis, takes dimension as required
    wt_shapes = []
    split_sizes = []
    for ii in np.arange(n_layers_hidden+1):
        if ii == 0:
            curshape = (-1, hidden_dim, inputs_dim)
        elif ii == n_layers_hidden:
            curshape = (-1, outputs_dim, hidden_dim)
        else:
            curshape = (-1, hidden_dim, hidden_dim)
        wt_shapes.append(curshape)           #weight matrix shape
        wt_shapes.append((-1,curshape[1]))   #bias vector shape

        cursize = curshape[1] * curshape[2]
        split_sizes.append(cursize)
        split_sizes.append(curshape[1])
    n_wt_elem = len(split_sizes)    #number  of weight elements

    weights_info = {'weights_total_dim'     : wt_total_dim,
                    'weights_shapes'        : wt_shapes,
                    'weights_split_sizes'   : split_sizes,
                    'n_weight_elements'     : n_wt_elem
                    }
    return weights_info

class keras_oneshot_v2(keras_oneshot_v1):
    '''Override self.call to use efficient, batch by case indexing/forward pass'''
  
    def __init__(self,  prob_set    = None, 
                        hypernet    = None,
                        **kwargs):
        super().__init__(   prob_set = prob_set, 
                            hypernet = hypernet,
                            **kwargs)

    def call(self, x):
        return(self.call_model_efficient(x = x[0], mu = x[1]))
    
    def call_model_efficient(self, x, mu, training=False):
        '''Efficient forward pass of full hypernet model, allows for batch size >= 1
        Args:
            x   : Tensor, w/dimension (n_batch x n_spatial x n_inptus), main net input
            mu  : Tensor, w/dimension (n_batch x n_param), hypernet input
        
        Returns:
            output[-1]  : Tensor, w/dimension (n_batch, x n_spatial x n_output),
                final output from main network, w/weights generated by hypernetwork
        '''

        n_spatial = x.shape[1]
        #call hypernet
        weights = self.hypernet(mu, training=training)

        #split output, reshape into weight/bias matrices/vectors
        split_weights = tf.split(weights, self.split_sizes, axis=1)
        weights_reshaped = []
        for iShape, target_shape in enumerate(self.weights_shapes):
            cur_weight = tf.reshape(split_weights[iShape], target_shape)
            weights_reshaped.append(cur_weight)

        #forward propagate the main network
        output  = []
        output.append(x)
        for iLayer in np.arange(self.model_opt['n_layers_hidden_main']+1):
            
            #count by 2's to extract weight, bias from weights_reshaped
            idx_start   = 2*iLayer
            Wcur        = weights_reshaped[idx_start]
            bcur        = weights_reshaped[idx_start + 1]

            #tile bias to allow batch size > 1
            bcur        = tf.expand_dims(bcur, axis=1)
            bcur        = tf.tile(bcur, tf.constant([1,n_spatial,1], tf.int32))

            #layer multiplication
            zh = tf.einsum('ijk,imk->imj', Wcur, output[iLayer]) + bcur
            hcur = self.activations[iLayer](zh)
            output.append(hcur)

        return output[-1]

########################################################################################################################

def fourier_v1_options_simple(  network_class           = None,
                                name                    = None,
                                n_layers_hidden_main    = None,
                                hidden_dim_main         = None,
                                inputs_dim_main         = None,
                                outputs_dim_main        = None,
                                n_fourier_features      = None,
                                fourier_gaussian_stddev = None,
                                inputs_dim_hypernet     = None,
                                outputs_dim_hypernet    = None,
                                activation_main         = None,
                                is_linear_output_main   = None,
                                inputs_main             = None,
                                inputs_hypernet         = None,
                                outputs_main            = None,
                                hypernet_opt            = None,
                                weights_info_main       = None,
                        ):
    '''Template for model options for keras_oneshot_v1
    Compute the main network weight shapes if not supplied, assuming constant 
    hidden dimension through the main network
    '''

    if weights_info_main is None:
        weights_info_main = v1_weight_dimensions_simple(n_layers_hidden = n_layers_hidden_main,
                                                        hidden_dim      = hidden_dim_main,
                                                        inputs_dim      = int(2 *n_fourier_features),
                                                        outputs_dim     = outputs_dim_main)
        outputs_dim_hypernet = weights_info_main['weights_total_dim']
    model_opt = locals()
    return model_opt
    

class keras_oneshot_fourier_v1(keras_oneshot_v2):
    '''Use a main network with a Fourier input layer, shared across all main networks.

    Use with efficient training, batch_size >=1, subclass keras_oneshot_v2 and override:
        - self.call_model_efficient
        - self.compose_main_network

    May still use v1_weight_dimensions_simple(), but pass the gaussian_projection dimension
        instead of the inputs dimension. This is handled automatically when using 
        fourier_v1_options_simple()
    '''
    def __init__(self, prob_set, hypernet, **kwargs):
        super().__init__(   prob_set    = prob_set,
                            hypernet    = hypernet,
                            **kwargs)
        
        #get the precision policy for building Fourier layer 
        self.global_policy = tf.keras.mixed_precision.global_policy()
        # self.build_fourier_layer(dtype = self.global_policy.compute_dtype)
        self.build_fourier_layer()
        
    def build_fourier_layer(self,dtype = None):
        '''Create the fourier layer which used for all predictions, and 
         explicitly  inserted into each main network when composed
         '''
        
        self.fourier_layer = FourierFeatureProjection(  n_features      = self.model_opt['n_fourier_features'],
                                                        gaussian_stddev = self.model_opt['fourier_gaussian_stddev'],)
        
    def call_model_efficient(self, x, mu, training=False):
        '''Efficient forward pass of full hypernet model, allows for batch size >= 1
        Args:
            x   : Tensor, w/dimension (n_batch x n_spatial x n_inptus), main net input
            mu  : Tensor, w/dimension (n_batch x n_param), hypernet input
        
        Returns:
            output[-1]  : Tensor, w/dimension (n_batch, x n_spatial x n_output),
                final output from main network, w/weights generated by hypernetwork
        '''

        n_spatial = x.shape[1]
        #call hypernet
        weights = self.hypernet(mu, training=training)

        #split output, reshape into weight/bias matrices/vectors
        split_weights = tf.split(weights, self.split_sizes, axis=1)
        weights_reshaped = []
        for iShape, target_shape in enumerate(self.weights_shapes):
            cur_weight = tf.reshape(split_weights[iShape], target_shape)
            weights_reshaped.append(cur_weight)

        #forward propagate the main network
        output  = []
        ff_out = self.fourier_layer(x)
        output.append(ff_out)
        for iLayer in np.arange(self.model_opt['n_layers_hidden_main']+1):
            
            #count by 2's to extract weight, bias from weights_reshaped
            idx_start   = 2*iLayer
            Wcur        = weights_reshaped[idx_start]
            bcur        = weights_reshaped[idx_start + 1]

            #tile bias to allow batch size > 1
            bcur        = tf.expand_dims(bcur, axis=1)
            bcur        = tf.tile(bcur, tf.constant([1,n_spatial,1], tf.int32))

            #layer multiplication
            zh = tf.einsum('ijk,imk->imj', Wcur, output[iLayer]) + bcur
            hcur = self.activations[iLayer](zh)
            output.append(hcur)

        return output[-1]
    
    def compose_main_network(self, mu):
        '''Given a design variable vector mu, generate the main network weights using
        the hypernetwork, load into Keras model and return
        
        Args:
            mu (tensor or ndarray)  : design variable vector

        Returns:
            model (tf.keras.Model)  : composed main network, w/generated weights
        '''
        weights         = self.hypernet(mu)
        n_layers_total  = self.model_opt['n_layers_hidden_main'] + 1

        #split output, reshape into weight/bias matrices/vectors
        split_weights = tf.split(weights, self.split_sizes, axis=1)
        wt_shp = []
        wt_squeeze = []
        for iShape, target_shape in enumerate(self.weights_shapes):
            cur_weight = tf.reshape(split_weights[iShape], target_shape)
            wt_shp.append(cur_weight)

            wt_sq = np.squeeze(cur_weight, axis=0)
            wt_squeeze.append(wt_sq.T)

        #make list of lists, gathering weights/biases by layer
        #Example: wts_all[0] = (W0, b0)
        wts_all = []
        for iLayer in np.arange(n_layers_total):
            idx1 = iLayer*2
            idx2 = idx1 + 1    
            curlayer_wt = [wt_squeeze[idx1], wt_squeeze[idx2]]
            wts_all.append(curlayer_wt)

        #build the main network as a tf.keras.Model
        #input layer and fourier projection
        input1 = tf.keras.Input(shape=(self.model_opt['inputs_dim_main']), name='input')
        output = self.fourier_layer(input1)
        
        # Dense Layers Construction
        print('Constructing Dense Layers')
        for iLayer in range(n_layers_total):
            layername = 'dense_{:1.0f}'.format(iLayer)
            print(layername)

            activ = self.activations[iLayer]
            if iLayer == (n_layers_total -1):
                units = self.model_opt['outputs_dim_main']
            else:
                units = self.model_opt['hidden_dim_main']

            output = tf.keras.layers.Dense(units         = units,
                                            activation   = activ,
                                            name         = layername)(output)
        
        model = tf.keras.Model(inputs=[input1], outputs=output)
        model.summary()

        model_weights = model.get_weights()

        #load the weights into the model
        for iLayer, curwts in enumerate(wts_all):
            curlayer = model.get_layer(index = iLayer+2) #iLayer+2 to skip input and fourier layers
            curlayer.set_weights(curwts)

        return model