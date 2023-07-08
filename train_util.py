
def train_opt_xy(training_fraction              = None,
                    epochs                      = None,
                    batch_size                  = None,
                    loss                        = None,
                    optimizer                   = None,
                    learning_rate               = None,
                    is_learning_rate_decay      = None,
                    is_learning_rate_decay_type = None,
                    const_epochs                = None,
                    const_steps                 = None,
                    decay_epochs                = None,
                    decay_steps                 = None,
                    decay_rate                  = None,
                    kernel_regularizer          = None,
                    lambda_l1                   = None,
                    lambda_l2                   = None,
                    x_norm_method               = None,
                    y_norm_method               = None,
                    is_profile                  = None,
                    histogram_freq              = None,
                    profile_batch               = None):
    '''Get training options when there is 1 input tensor (x) and 1 output tensor (y),
    regardless of shape/dimension of inputs/outputs'''
    train_opt = locals()

    return train_opt

def train_opt_xmuy( training_fraction           = None,
                    epochs                      = None,
                    batch_size                  = None,
                    loss                        = None,
                    optimizer                   = None,
                    learning_rate               = None,
                    is_learning_rate_decay      = None,
                    learning_rate_decay_type    = None,
                    const_epochs                = None,
                    const_steps                 = None,
                    decay_epochs                = None,
                    decay_steps                 = None,
                    decay_rate                  = None,
                    kernel_regularizer          = None,
                    lambda_l1                   = None,
                    lambda_l2                   = None,
                    x_norm_method               = None,
                    mu_norm_method              = None,
                    y_norm_method               = None,
                    is_profile                  = None,
                    histogram_freq              = None,
                    profile_batch               = None):
    '''Get dict of all required training options when there are 2 input tensors
    (x, mu) and 1 output tensor (y), regardless of shape/dimension of 
    inputs/outputs
    '''
    
    train_opt = locals()
    
    return train_opt