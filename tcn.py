import tensorflow as tf
from conv_tasnet import ConvTasNetParam


class gLN(tf.keras.layers.Layer):

    """Global Layer Normalization"""

    __slots__ = 'eps'  # small constant for numerical stability

    def __init__(self, eps: float = 1e-8, **kwargs):
        super(gLN, self).__init__(**kwargs)
        self.eps = eps

    def call(self, inputs):
        pass
# gLN end


class cLN(tf.keras.layers.Layer):

    """Cumulative Layer Normalization"""

    __slots__ = 'eps'  # small constant for numerical stability

    def __init__(self, eps: float = 1e-8, **kwargs):
        super(cLN, self).__init__(**kwargs)
        self.eps = eps

    def call(self, inputs):
        pass
# cLN end


class Conv1DBlock(tf.keras.layers.Layer):

    """1-D Dilated Convolutional Block"""

    __slots__ = ('param', 'dilation',
                 'conv1x1', 'prelu1', 'normalization1',
                 'dconv', 'prelu2', 'normalization2',
                 'conv1x1_B', 'conv1x1_Sc')

    def __init__(self, param: ConvTasNetParam, dilation: int = 1, **kwargs):
        super(Conv1DBlock, self).__init__(**kwargs)
        self.param = param
        self.dilation = dilation

        self.conv1x1 = tf.keras.layers.Conv1D(
            filters=self.param.H, kernel_size=1, use_bias=False)
        self.prelu1 = tf.keras.layers.PReLU()  # for pointwise convolution (1x1-conv)

        self.dconv = tf.keras.layers.Conv1D(filters=self.param.H, padding='same',
                                            use_bias=False, dilation_rate=self.dilation)
        self.prelu2 = tf.keras.layers.PReLU()  # for depthwise convolution (D-conv)

        self.conv1x1_B = tf.keras.layers.Conv1D(
            filters=self.param.B, kernel_size=1, use_bias=False)
        self.conv1x1_Sc = tf.keras.layers.Conv1D(
            filters=self.param.Sc, kernel_size=1, use_bias=False)

        if(self.param.causality):
            self.normalization1 = cLN(eps=self.param.eps)  # for 1x1-conv
            self.normalization2 = cLN(eps=self.param.eps)  # for D-conv
        else:
            self.normalization1 = gLN(eps=self.param.eps)  # for 1x1-conv
            self.normalization2 = gLN(eps=self.param.eps)  # for D-conv

    def call(self, inputs):
        """
        Args:
            inputs: [T_hat x B]

        Returns:
            outputs: [T_hat x B]
            skipconnection: [T_hat x Sc]
        """
        # TODO | Add skip connection path and residual path
        outputs = self.conv1x1(inputs)  # []
        outputs = self.prelu1(outputs)
        outputs = self.normalization1(outputs)

        outputs = self.dconv(outputs)
        outputs = self.prelu2(outputs)
        outputs = self.normalization2(outputs)

        return outputs
# Conv1DBlock end


class TCN(tf.keras.layers.Layer):

    """Dilated Temporal Convolutional Network"""

    __slots__ = ('param', 'stack')

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(TCN, self).__init__(**kwargs)
        self.param = param
        self.stack = []  # stack of the Conv1DBlock instances
        # TODO | stack Conv1DBlock into self.stack

    def call(self, tcn_inputs):
        """
        Args:
            tcn_inputs: [T_hat x B]

        Returns:
            tcn_outputs: [T_hat x B]
        """
        pass
# TCN end
