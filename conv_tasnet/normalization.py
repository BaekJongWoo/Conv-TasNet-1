import tensorflow as tf


class GlobalLayerNorm(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(GlobalLayerNorm, self).__init__(name='gLN', **kwargs)
        self.eps = 1e-7

    def build(self, input_shape):
        self.g = self.add_weight(name='gLN_gamma',
                                 shape=(int(input_shape[-1]), ),
                                 initializer='glorot_uniform',
                                 trainable=True)

        self.b = self.add_weight(name='gLN_beta',
                                 shape=(int(input_shape[-1]), ),
                                 initializer='glorot_uniform',
                                 trainable=True,)

    def call(self, inputs):
        m = tf.math.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        v = tf.math.reduce_variance(inputs, axis=[1, 2], keepdims=True)
        return ((inputs - m) / tf.math.sqrt(v + self.eps)) * self.g + self.b


class CausalLayerNorm(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(CausalLayerNorm, self).__init__(name='cLN', **kwargs)
        self.eps = 1e-7

    def build(self, input_shape):
        self.K = input_shape[-2]
        self.g = self.add_weight(name='cLN_gamma',
                                 shape=(int(input_shape[-1]), ),
                                 initializer='glorot_uniform',
                                 trainable=True)

        self.b = self.add_weight(name='cLN_beta',
                                 shape=(int(input_shape[-1]), ),
                                 initializer='glorot_uniform',
                                 trainable=True)

    def call(self, inputs):
        # k_count: number of frames that have to be taken in k-th frame of the inputs
        k_count = tf.cast(tf.reshape(range(1, self.K + 1),
                          [1, self.K, 1]), tf.float32)

        # k_mean: mean of N entries for each frame of the inputs
        k_mean = tf.math.reduce_mean(inputs, axis=-1, keepdims=True)
        # k_mean: mean of N entries square for each frame of the inputs
        k_pow_mean = tf.math.reduce_mean(
            tf.math.pow(inputs, 2), axis=-1, keepdims=True)

        k_sum = tf.math.cumsum(k_mean, axis=-2)
        k_pow_sum = tf.math.cumsum(k_pow_mean, axis=-2)

        m = k_sum/k_count
        v = (k_pow_sum - 2*k_mean*k_sum)/k_count + tf.math.pow(k_mean, 2)
        return ((inputs - m) / tf.math.sqrt(v + self.eps)) * self.g + self.b
