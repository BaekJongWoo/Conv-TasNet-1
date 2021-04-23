import tensorflow as tf


class GlobalLayerNorm(tf.keras.layers.Layer):

    def __init__(self, H, **kwargs):
        super(GlobalLayerNorm, self).__init__(name='gLN', **kwargs)
        self.eps = tf.keras.backend.epsilon()

    def build(self, input_shape):
        _shape = (int(input_shape[-1]), )
        self.g = self.add_weight(name='gLN_gamma',
                                 shape=_shape,
                                 initializer='glorot_uniform',
                                 trainable=True)

        self.b = self.add_weight(name='gLN_beta',
                                 shape=_shape,
                                 initializer='glorot_uniform',
                                 trainable=True,)

    def call(self, inputs):
        m = tf.math.reduce_mean(inputs, axis=[-2, -1], keepdims=True)
        v = tf.math.reduce_variance(inputs, axis=[-2, -1], keepdims=True)
        outputs = ((inputs - m) / tf.math.sqrt(v + self.eps)) * self.g + self.b
        return outputs


class CausalLayerNorm(tf.keras.layers.Layer):

    def __init__(self, H, **kwargs):
        super(CausalLayerNorm, self).__init__(name='cLN', **kwargs)
        self.eps = tf.keras.backend.epsilon()

    def build(self, input_shape):
        _shape = (int(input_shape[-1]), )
        self.g = self.add_weight(name='cLN_gamma',
                                 shape=_shape,
                                 initializer='glorot_uniform',
                                 trainable=False)

        self.b = self.add_weight(name='cLN_beta',
                                 shape=_shape,
                                 initializer='glorot_uniform',
                                 trainable=False)

    def call(self, inputs):
        m = tf.math.reduce_mean(inputs, axis=[-2, -1], keepdims=True)
        v = tf.math.reduce_variance(inputs, axis=[-2, -1], keepdims=True)
        outputs = ((inputs - m) / tf.math.sqrt(v + self.eps)) * self.g + self.b
        return outputs
