import tensorflow as tf
from conv_tasnet import ConvTasNetParam


class gLN(tf.keras.layers.Layer):

    """Global Layer Normalization"""

    __slots__ = ('eps')

    def __init__(self, eps: float = 1e-8, **kwargs):
        super(gLN, self).__init__(**kwargs)
        self.eps = eps  # small constant for numerical stability

    def call(self):
        pass
# gLN end


class cLN(tf.keras.layers.Layer):

    """Cumulative Layer Normalization"""

    __slots__ = ('eps')

    def __init__(self, eps: float = 1e-8, **kwargs):
        super(cLN, self).__init__(**kwargs)
        self.eps = eps  # small constant for numerical stability

    def call(self):
        pass
# cLN end


class Conv1DBlock(tf.keras.layers.Layer):

    __slots__ = ('param', 'prelu1', 'prelu2')

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(Conv1DBlock, self).__init__(**kwargs)
        self.param = param
        self.prelu1 = tf.keras.layers.PReLU()  # for pointwise convolution (1x1-conv)
        self.prelu2 = tf.keras.layers.PReLU()  # for depthwise convolution (D-conv)

    def call(self):
        pass
# Conv1DBlock end


class TCN(tf.keras.layers.Layer):

    """Dilated Temporal Convolutional Network"""

    __slots__ = ('param')

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(TCN, self).__init__(**kwargs)
        self.param = param

    def call(self):
        pass
# TCN end
