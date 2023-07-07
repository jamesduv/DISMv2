import tensorflow as tf
from tensorflow.keras import mixed_precision

def swish(x):
    return (x*tf.keras.activations.sigmoid(x))
tf.keras.utils.get_custom_objects().update({'swish' : swish})  

def set_tensorflow_precision_policy(    backend_precision_value     = 'float64',   
                                        is_mixed_precision          = False,):
    '''Set tensorflow backend precision policy'''
    
    if is_mixed_precision:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    else:
        tf.keras.backend.set_floatx(backend_precision_value)

def get_loss(target_loss):
    all_losses = {'KLD'     : tf.keras.losses.KLD,
                  'MAE'     : tf.keras.losses.MAE,
                  'MAPE'    : tf.keras.losses.MAPE,
                  'MSE'     : tf.keras.losses.MSE,
                  'mse'     : tf.keras.losses.MSE,
                  'MSLE'    : tf.keras.losses.MSLE,
                  'binary_crossentropy' : tf.keras.losses.binary_crossentropy,
                  'categorical_crossentropy'    : tf.keras.losses.categorical_crossentropy,
                  'categorical_hinge'   : tf.keras.losses.categorical_hinge}

    assert target_loss in all_losses.keys()
    return all_losses[target_loss]

def get_activation(target_activ, **kwargs):

    activations = {'elu'                : tf.keras.activations.elu,
                    'hard_sigmoid'      : tf.keras.activations.hard_sigmoid,
                    'linear'            : tf.keras.activations.linear,
                    'relu'              : tf.keras.activations.relu,
                    'selu'              : tf.keras.activations.selu,
                    'sigmoid'           : tf.keras.activations.sigmoid,
                    'softmax'           : tf.keras.activations.softmax,
                    'tanh'              : tf.keras.activations.tanh,
                    'swish'             : swish,}

    assert target_activ in activations.keys()
   
    return activations[target_activ]

def get_optimizer_handle(target_optim):
    optimizers = {  'Adadelta'  : tf.keras.optimizers.Adadelta,
                    'Adagrad'   : tf.keras.optimizers.Adagrad,
                    'Adam'      : tf.keras.optimizers.Adam,
                    'Adamax'    : tf.keras.optimizers.Adamax,
                    'Ftrl'      : tf.keras.optimizers.Ftrl,
                    'Nadam'     : tf.keras.optimizers.Nadam,
                    'RMSprop'   : tf.keras.optimizers.RMSprop,
                    'SGD'       : tf.keras.optimizers.SGD
                    }
    assert target_optim in optimizers.keys()
    return optimizers[target_optim]