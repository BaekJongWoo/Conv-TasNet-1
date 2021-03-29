import tensorflow as tf

class CLN(tf.keras.layers.Layer):

    """Casual Layer Normalization for Casual TCN"""

    def __init__(self, **kwargs):
        pass

    def call(self, **kwargs):
        pass

class GLN(tf.keras.layers.Layer):

    """Global Layer Normalization for Non-Casual TCN"""

    def __init__(self, **kwargs):
        pass

    def call(self, **kwargs):
        pass

class TCN(tf.keras.layers.Layer):

    """Dilated Temporal Convolution Network"""

    def __init__(self, param, **kwargs):
        self.param = param

    def call(self, tcn_inputs):
        pass
