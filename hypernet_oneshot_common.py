import os

import dense_networks
import hypernet_oneshot_networks as hypernet

def get_f_model(opt):
    '''Get the function handles for the classes defined above
    
    Args:
        opt (dict)  : net settings
    
    Returns:
        f_model (handle) : model class handle
    '''
    network_class = opt['network_class']
    f_allowed = {   'keras_oneshot_v1'          : hypernet.keras_oneshot_v1,
                    'keras_oneshot_v2'          : hypernet.keras_oneshot_v2,
                    'keras_oneshot_fourier_v1'  : hypernet.keras_oneshot_fourier_v1,
                 }
    if network_class not in f_allowed.keys():
        raise Exception('Unsupported model class: {}'.format(network_class) )
    f_model = f_allowed[network_class]
    return f_model

def get_f_dense(opt):
    '''Get the function handle for the dense hypernetwork, from dense_networks.py
    
    Args:
        opt (dict)  : net settings
    
    Returns:
        f_model (handle) : model class handle
    '''

    f_allowed = {'dense_hypernet_v1' : dense_networks.dense_hypernet_v1}
    if opt['network_class'] not in f_allowed.keys():
        raise Exception('Unsupported model class: {}'.format(opt['network_class']))
    f_model = f_allowed[opt['network_class']]
    return f_model

def get_dense_hypermodel(prob_set):
    '''Get and return an oveall hypermodel w/ dense hypernetwork'''
    
    opt = prob_set.opt

    #get the dense hypernetwork
    f_dense     = get_f_dense(opt['model_opt']['hypernet_opt'])
    dense_class = f_dense(opt = opt['model_opt']['hypernet_opt'])
    dense_class.build_model()
    hypernet    = dense_class.model

    #get the overall hypernetwork model
    f_model     = get_f_model(opt['model_opt'])
    net         = f_model(prob_set, hypernet)

    return net, dense_class