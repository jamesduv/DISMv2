import tensorflow as tf 
import numpy as np

# modified from https://github.com/titu1994/tf_fourier_features
class FourierFeatureProjection(tf.keras.layers.Layer):
    def __init__(self, gaussian_projection, gaussian_scale, **kwargs):
        super().__init__(**kwargs)

        self.gauss_proj     = gaussian_projection
        self.gauss_scale    = gaussian_scale

    def build(self, input_shape):
        input_dim = input_shape[-1]

        if self.gauss_proj <= 0:
            # Assume basic projection
            self.proj_kernel = tf.keras.layers.Dense(   input_dim, 
                                                        use_bias    = False, 
                                                        trainable   = False,
                                                        kernel_initializer = 'identity', )

        else:
            initializer = tf.keras.initializers.TruncatedNormal(mean    = 0.0, 
                                                                stddev  = self.gauss_scale)
            self.proj_kernel = tf.keras.layers.Dense(units      = self.gauss_proj,
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
            'gaussian_projection'   : self.gauss_proj,
            'gaussian_scale'        : self.gauss_scale
                }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))