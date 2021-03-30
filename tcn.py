import tensorflow as tf
from conv_tasnet import ConvTasNetParam


class cLN(tf.keras.layers.Layer):

    __slots__ = ('eps')

    def __init__(self, eps: float = 1e-8, **kwargs):
        super(cLN, self).__init__(**kwargs)
        self.eps = eps  # small constant for numerical stability

    def call(self, inputs):
        pass
# cLN end


class Conv1DBlock(tf.keras.layers.Layer):

    __slots__ = ('param', 'is_causal', 'eps',
                 'conv1x1_reshape1', 'conv1x1', 'conv1x1_reshape2',
                 'prelu1', 'normalization1',
                 'dconv_reshape1', 'dconv', 'dconv_reshape2',
                 'prelu2', 'normalization2')

    def __init__(self, param: ConvTasNetParam, is_causal: bool = True, eps: float = 1e-8, **kwargs):
        super(Conv1DBlock, self).__init__(**kwargs)
        self.param = param
        self.is_causal = is_causal
        self.eps = eps  # normalization (for numerical stability)
        self.conv1x1_reshape1 = tf.keras.layers.Reshape()  # TODO | Add target_shape
        self.conv1x1 = tf.keras.layers.Conv2D()  # TODO | filters, kernel_size, padding
        self.conv1x1_reshape2 = tf.keras.layers.Reshape()  # TODO | Add target_shape
        self.prelu1 = tf.keras.layers.PReLU()  # for pointwise convolution (1x1-conv)
        self.dconv_reshape1 = tf.keras.layers.Reshape()  # TODO | Add target_shape
        self.dconv = tf.keras.layers.Conv2D()  # TODO | filters, kernel_size, padding
        self.dconv_reshape2 = tf.keras.layers.Reshape()  # TODO | Add target_shape
        self.prelu2 = tf.keras.layers.PReLU()  # for depthwise convolution (D-conv)

        # TODO | implement skip connection and residual layer

        if(self.is_causal):
            self.normalization1 = cLN(eps=1e-8)
            self.normalization2 = cLN(eps=1e-8)
        else:
            self.normalization1 = tf.keras.layers.LayerNormalization(
                epsilon=1e-8)
            self.normalization2 = tf.keras.layers.LayerNormalization(
                epsilon=1e-8)

    def call(self, inputs):
        # TODO | implement skip connection and residual path
        outputs = self.conv1x1_reshape1(inputs)
        outputs = self.conv1x1(outputs)
        outputs = self.conv1x1_reshape2(outputs)
        outputs = self.prelu1(outputs)
        outputs = self.normalization1(outputs)

        outputs = self.dconv_reshape1(outputs)
        outputs = self.dconv(outputs)
        outputs = self.dconv_reshape2(outputs)
        outputs = self.prelu2(outputs)
        outputs = self.normalization2(outputs)

        return outputs
# Conv1DBlock end


class TCN(tf.keras.layers.Layer):

    """Dilated Temporal Convolutional Network"""

    __slots__ = ('param', 'is_causal', 'eps', 'stack')

    def __init__(self, param: ConvTasNetParam, is_causal: bool = True, eps: float = 1e-8, **kwargs):
        super(TCN, self).__init__(**kwargs)
        self.param = param
        self.is_causal = is_causal
        self.eps = eps  # for normalization (numerical stability)
        self.stack = []  # stack of the Conv1DBlock instances
        # TODO | stack Conv1DBlock into self.stack

    def call(self, tcn_inputs):
        pass
# TCN end
