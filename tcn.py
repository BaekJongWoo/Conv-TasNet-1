import tensorflow as tf
from config import ConvTasNetParam


class GlobalLayerNorm(tf.keras.layers.Layer):
    """Global Layer Normalization (i.e., gLN)

    Attributes:
        gamma (tf.Variable): Trainable parameter
        beta (tf.Varaible): Trainable paramter
        epsilon (float): Small constant for numerical stability
    """

    __slots__ = ("gamma", "beta", "epsilon")

    def __init__(self, eps: float = 1e-8, **kwargs):
        super(GlobalLayerNorm, self).__init__(**kwargs)
        self.epsilon = eps
        self.gamma = tf.Variable(trainable=True)
        self.beta = tf.Variable(trainable=True)

    def call(self, inputs):
        pass
# GlobalLayerNorm end


class CumulativeLayerNorm(tf.keras.layers.Layer):
    """Cumulative Layer Normalization (i.e., cLN)

    Attributes:
        gamma (tf.Variable): Trainable parameter
        beta (tf.Varaible): Trainable paramter
        epsilon (float): Small constant for numerical stability
    """

    __slots__ = ("gamma", "beta", "epsilon")

    def __init__(self, eps: float = 1e-8, **kwargs):
        super(CumulativeLayerNorm, self).__init__(**kwargs)
        self.epsilon = eps
        self.gamma = tf.Variable(trainable=True)
        self.beta = tf.Variable(trainable=True)

    def call(self, inputs):
        pass
# CumulativeLayerNorm end


class Conv1DBlock(tf.keras.layers.Layer):
    """1-D Convolution Block using Depthwise Separable Convolution

    TODO:
        Add causality depended layer normalization for pointwise_conv, and depthwsie_conv respectively

    Attributes
        param (ConvTasNetParam): Hyperparamters
        dilation (int): Dilation factor
    """

    def __init__(self, param: ConvTasNetParam, dilation: int, **kwargs):
        super(Conv1DBlock, self).__init__(**kwargs)
        self.param = param
        self.dilation = dilation
        self.reshape1 = tf.keras.layers.Reshape(
            target_shape=(self.param.T_hat, self.param.B, 1))
        self.pointwise_conv = tf.keras.layers.Conv2D(filters=self.param.H,
                                                     kernel_size=(
                                                         1, self.param.B),
                                                     use_bias=False)
        self.reshape2 = tf.keras.layers.Reshape(
            target_shape=(self.param.T_hat, self.param.H))
        self.prelu1 = tf.keras.layers.PReLU()
        self.reshape3 = tf.keras.layers.Reshape(
            target_shape=(self.param.T_hat, self.param.H, 1))
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(self.param.P, self.param.P),
                                                              dilation_rate=self.dilation,
                                                              padding="same",
                                                              use_bias=False)
        self.prelu2 = tf.keras.layers.PReLU()
        self.residual_conv1x1 = tf.keras.layers.Conv2D(filters=self.param.B,
                                                       kernel_size=(
                                                           1, self.param.H),
                                                       use_bias=False)
        self.reshape4 = tf.keras.layers.Reshape(
            target_shape=(self.param.T_hat, self.param.B))
        self.skipconn_conv1x1 = tf.keras.layers.Conv2D(filters=self.param.Sc,
                                                       kernel_size=(
                                                           1, self.param.H),
                                                       use_bias=False)
        self.reshape5 = tf.keras.layers.Reshape(
            target_shape=(self.param.T_hat, self.param.Sc))

    def call(self, block_inputs):
        """
        Args:
            block_inputs: (, T_hat, B)

        Returns:
            residual_outputs: (, T_hat, B)
            skipconn_outputs: (, T_hat, Sc)
        """
        # (, T_hat, B) -> (, T_hat, B, 1)
        pointwise_inputs = self.reshape1(block_inputs)
        # (, T_hat, B, 1) -> (, T_hat, H, 1)
        pointwise_outputs = self.pointwise_conv(pointwise_inputs)
        # (, T_hat, H, 1) -> (, T_hat, H)
        pointwise_outputs = self.reshape2(pointwise_outputs)
        # (, T_hat, H) -> (, T_hat, H)
        pointwise_outputs = self.prelu1(pointwise_outputs)
        # (, T_hat, H) -> (, T_hat, H, 1)
        pointwise_outputs = self.reshape3(pointwise_outputs)
        # (, T_hat, H, 1) -> (, T_hat, H, 1)
        depthwise_outputs = self.depthwise_conv(pointwise_outputs)
        # (, T_hat, H, 1) -> (, T_hat, H)
        depthwise_outputs = self.reshape2(depthwise_outputs)
        # (, T_hat, H) -> (, T_hat, H)
        depthwise_outputs = self.prelu2(depthwise_outputs)
        # (, T_hat, H) -> (, T_hat, H, 1)
        depthwise_outputs = self.reshape3(depthwise_outputs)
        # (, T_hat, H, 1) -> (, T_hat, B, 1)
        residual_outputs = self.residual_conv1x1(depthwise_outputs)
        # (, T_hat, B, 1) -> (, T_hat, B)
        residual_outputs = self.reshape4(residual_outputs)
        # (, T_hat, B), (, T_hat, B) -> (, T_hat, B)
        residual_outputs = residual_outputs + block_inputs
        # (, T_hat, H, 1) -> (, T_hat, Sc, 1)
        skipconn_outputs = self.skipconn_conv1x1(depthwise_outputs)
        # (, T_hat, Sc, 1) -> (, T_hat, Sc)
        skipconn_outputs = self.reshape5(skipconn_outputs)
        return residual_outputs, skipconn_outputs

    def get_config(self):
        return {**self.param.get_config(),
                "Dilation": self.dilation}
# Conv1DBlock end


class TemporalConvNet(tf.keras.layers.Layer):
    """Dilated Temporal Convolution Network (Dilated-TCN)

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        conv1dblock_stack (List[Conv1DBlock]): Dilated causal/noncausal depthwise separable network
    """

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(TemporalConvNet, self).__init__(**kwargs)
        self.param = param
        self.conv1dblock_stack = []
        for _ in range(self.param.R):
            for x in range(self.param.X):
                self.conv1dblock_stack.append(Conv1DBlock(self.param, 2**x))

    def call(self, tcn_inputs):
        """
        Args:
            tcn_inputs: (, T_hat, B)

        Locals:
            skipconn_outputs: (, T_hat, Sc)

        Returns:
            tcn_outputs: (, T_hat, Sc)
        """
        tcn_outputs = tf.zeros(shape=(self.param.T_hat, self.param.Sc))
        for block in self.conv1dblock_stack:
            residual_outputs, skipconn_outputs = block(tcn_inputs)
            tcn_inputs = residual_outputs
            tcn_outputs += skipconn_outputs
        return tcn_outputs

    def get_config(self):
        return self.param.get_config()
# TemporalConvNet end
