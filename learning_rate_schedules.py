import tensorflow as tf

class piecewiseConstantExpDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    '''Compute the step size for a piecewise constant-exponential decay schedule, given the optimizer step.
    The step size is computed as:

        if step_i <= n_steps_const
            lr_i = lr_0
        else:
            lr_i = lr_0 * decay_rate ^ [ (step - n_steps_const) / n_steps_decay ]

    This means the step size decays by a factor of decay_rate every n_steps_decay, after a period of constant lr_0 
    training.

    Given the settings, convert from epochs to steps for (y) as:

        n_steps_(y) = n_epochs_(y) x n_batches_per_epoch 
    '''
    def __init__(self,  initial_learning_rate,
                        n_steps_const,
                        n_steps_decay,
                        decay_rate):


        #get the precision policy
        self.global_policy = tf.keras.mixed_precision.global_policy()

        #cast once here instead of on each call of const_lr()
        self.initial_learning_rate  = tf.cast(initial_learning_rate, self.global_policy.variable_dtype)
        self.n_steps_const          = n_steps_const
        self.n_steps_decay          = n_steps_decay
        self.decay_rate             = tf.cast(decay_rate, self.global_policy.variable_dtype)
        self.step                   = 0


    def __call__(self, step):
        self.step = step
        #condition, return true_fn when in period of constant lr, return false_fn otherwise
        cur_learning_rate = tf.cond(    step <= self.n_steps_const, 
                                        true_fn     = self.const_lr, 
                                        false_fn    = self.exp_lr )
        return cur_learning_rate

    def const_lr(self,):
        '''Callable to return the initial learning rate, for constant period of training'''
        return self.initial_learning_rate

    def exp_lr(self,):
        '''Callable to return the decayed learning rate, for period of exponential decay
        l_new = l_init * decay_rate ^ [ (step - n_steps_const) / n_steps_decay]'''
        expon   = (self.step - self.n_steps_const) / (self.n_steps_decay)
        expon   = tf.cast(expon,self.global_policy.variable_dtype )
        new_lr  = self.initial_learning_rate*(self.decay_rate**expon)
        return tf.cast(new_lr, self.global_policy.variable_dtype)   