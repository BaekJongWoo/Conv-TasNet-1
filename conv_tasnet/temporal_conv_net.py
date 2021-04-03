import tensorflow as tf
from .conv_tasnet_param import ConvTasNetParam
from .normalizations import GlobalLayerNorm as gLN
from .normalizations import CumulativeLayerNorm as cLN


class ConvTasNetTCN(tf.keras.layers.Layer):
    """Dilated Temporal Convolutional Network

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        conv1d_block_list (List[ConvTasNetConv1DBlock]): List of 1-D convolution blocks
    """

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetTCN, self).__init__(**kwargs)
        self.param = param
        self.conv1d_block_list = []
        for _ in range(self.param.R):  # for each repeat
            for x in range(self.param.X):
                self.conv1d_block_list.append(
                    ConvTasNetConv1DBlock(self.param, dilation=2**x))
        # avoid gradient missing warning
        self.conv1d_block_list[-1].is_last = True

    def call(self, tcn_inputs: tf.Tensor) -> tf.Tensor:
        """
        Args:
            tcn_input (tf.Tensor): Tensor of shape=(, K, B)

        Returns:
            tcn_output (tf.Tensor): Tensor of shape=(, K, S)
        """
        tcn_outputs = tf.zeros(shape=(self.param.K, self.param.S))
        for block in self.conv1d_block_list:
            # (, K, B) -> (, K, B), (, K, S)
            residual_outputs, skipconn_outputs = block(tcn_inputs)
            # (, K, B) -> (, K, B)
            tcn_inputs = residual_outputs
            # (, K, S) -> (, K, S)
            tcn_outputs += skipconn_outputs
        return tcn_outputs

    def get_config(self) -> dict:
        return self.param.get_config()
# ConvTasNetTCN end


class ConvTasNetConv1DBlock(tf.keras.layers.Layer):
    """1-D Convolutional Block using Depthwise Separable Convolution

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        dilation (int): Dilation factor
        is_last (bool): Flag whether target instance is last block in TCN
        bottleneck_conv (tf.keras.layers.Conv1D): 1x1 convolution layer
        prelu1 (tf.keras.layers.PReLU): PReLU activation layer for bottleneck_conv
        normalization1 (cLN | gLN): Causality depended normalization layer
        depthwise_conv (tf.keras.layers.Conv1D): 1-D depthwise convolution layer
        prelu2 (tf.keras.layers.PReLU): PReLU activation layer for depthwise_conv
        normalization2 (cLN | gLN): Causality depended normalization layer
        residual_conv (tf.keras.layers.Conv1D): 1x1 convolution layer corresponding to the resodual path
        skipconn_conv (tf.keras.layers.Conv1D): 1x1 convolution layer corresponding to the skipconnection path
    """

    def __init__(self, param: ConvTasNetParam, dilation: int, **kwargs):
        super(ConvTasNetConv1DBlock, self).__init__(**kwargs)
        self.param = param
        self.dilation = dilation
        self.is_last = False

        if self.param.causal:  # causal system
            self.causal = "causal"
            self.normalization1 = cLN(H=self.param.H,
                                      eps=self.param.eps)
            self.normalization2 = cLN(H=self.param.H,
                                      eps=self.param.eps)
        else:  # noncausal system
            self.causal = "same"
            self.normalization1 = gLN(H=self.param.H,
                                      eps=self.param.eps)
            self.normalization2 = gLN(H=self.param.H,
                                      eps=self.param.eps)

        self.bottleneck_conv = tf.keras.layers.Conv1D(filters=self.param.H,
                                                      kernel_size=1,
                                                      use_bias=False)
        self.prelu1 = tf.keras.layers.PReLU()
        self.depthwise_conv = tf.keras.layers.Conv1D(filters=self.param.H,
                                                     kernel_size=self.param.P,
                                                     dilation_rate=self.dilation,
                                                     padding=self.param.causal,
                                                     groups=self.param.H,
                                                     use_bias=False)
        self.prelu2 = tf.keras.layers.PReLU()
        self.residual_conv = tf.keras.layers.Conv1D(filters=self.param.B,
                                                    kernel_size=1,
                                                    use_bias=False)
        self.skipconn_conv = tf.keras.layers.Conv1D(filters=self.param.Sc,
                                                    kernel_size=1,
                                                    use_bias=False)

    def call(self, block_inputs: tf.Tensor) -> tf.Tensor:
        """
        Args:
            block_inputs (tf.Tensor): Tensor of shape=(, K, B)

        Returns:
            residual_outputs (tf.Tensor): Tensor of shape=(, K, B)
            skipconn_outputs (tf.Tensor): Tensor of shape=(, K, S)
        """
        # (, K, B) -> (, K, H)
        bottleneck_outputs = self.bottleneck_conv(block_inputs)
        # (, K, H) -> (, K, H)
        depthwise_inputs = self.prelu1(bottleneck_outputs)
        # (, K, H) -> (, K, H)
        depthwise_inputs = self.normalization1(depthwise_inputs)
        # (, K, H) -> (, K, H)
        depthwise_outputs = self.depthwise_conv(depthwise_inputs)
        # (, K, H) -> (, K, H)
        depthwise_outputs = self.prelu2(depthwise_outputs)
        # (, K, H) -> (, K, H)
        depthwise_outputs = self.normalization2(depthwise_outputs)

        # avoid gradient missing
        # (, K, H) -> (, K, B)
        residual_outputs = block_inputs
        if not self.is_last:
            residual_outputs += self.residual_conv(depthwise_outputs)
        # (, K, H) -> (, K, S)
        skipconn_outputs = self.skipconn_conv(depthwise_outputs)

        return residual_outputs, skipconn_outputs

    def get_cofig(self) -> dict:
        return self.param.get_config()
# ConvTasNetConv1DBlock end