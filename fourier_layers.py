import tensorflow as tf 
import numpy as np

# modified from https://github.com/titu1994/tf_fourier_features
class FourierFeatureProjection(tf.keras.layers.Layer):
    def __init__(self, n_features, gaussian_stddev, **kwargs):
        super().__init__(**kwargs)

        assert gaussian_stddev > 0, 'A positive standard deviation must be specified, {:1.2e} received'.format(gaussian_stddev)
        self.n_features         = n_features
        self.gaussian_stddev    = gaussian_stddev

    def build(self, input_shape):
        input_dim = input_shape[-1]

        initializer = tf.keras.initializers.TruncatedNormal(    mean    = 0.0, 
                                                                stddev  = self.gaussian_stddev)
        self.proj_kernel = tf.keras.layers.Dense(   units      = self.n_features,
                                                    activation = 'linear',
                                                    use_bias   = False, 
                                                    trainable  = False,
                                                    kernel_initializer = initializer,)
        self.built = True

    def call(self, inputs, **kwargs):
        x_proj = 2.0 * np.pi * inputs
        x_proj = self.proj_kernel(x_proj)

        x_proj_sin = tf.sin(x_proj)
        x_proj_cos = tf.cos(x_proj)

        output = tf.concat([x_proj_sin, x_proj_cos], axis=-1)
        return output

    def get_config(self):
        config = {
            'n_features'        : self.n_features,
            'gaussian_scale'    : self.gaussian_stddev,
                }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))