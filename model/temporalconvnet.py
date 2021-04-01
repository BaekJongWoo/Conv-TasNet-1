import tensorflow as tf
from convtasnet_param import ConvTasNetParam


class Conv1DBlock(tf.keras.layers.Layer):
    """1-D Convolution Block using Depthwise Separable Convolution

    TODO:
        Add 'causality depended' layer normalization for bottleneck_conv, and depthwsie_conv respectively

    Attributes
        param (ConvTasNetParam): Hyperparamters
        dilation (int): Dilation factor
        bottleneck_conv (keras.layers.Conv1D): 1x1 convolution layer
        prelu1 (keras.layers.PReLU): PReLU activation layer for the bottleneck_conv layer
        depthwise_conv (keras.layers.Conv1D): 1-D depthwise convolution layer
        prelu2 (keras.layers.PReLU): PReLU activation layer for the depthwise_conv layer
        residual_conv (keras.layers.Conv1D): 1x1 convolution layer corresponding to the resodual path
        skipconn_conv (keras.layers.Conv1D): 1x1 convolution layer corresponding to the skipconnection path
    """

    def __init__(self, param: ConvTasNetParam, dilation: int, **kwargs):
        super(Conv1DBlock, self).__init__(**kwargs)
        self.param = param
        self.dilation = dilation
        self.bottleneck_conv = tf.keras.layers.Conv1D(filters=self.param.H,
                                                      use_bias=False)
        self.prelu1 = tf.keras.layers.PReLU()

        if self.param.causality:  # causal system
            self.causal = "causal"
            self.layer_normalization1 = CumulativeLayerNorm(self.param.T_hat)
            self.layer_normalization2 = CumulativeLayerNorm(self.param.T_hat)
        else:  # noncausal system
            self.causal = "same"
            self.layer_normalization1 = GlobalLayerNorm(self.param.T_hat)
            self.layer_normalization2 = GlobalLayerNorm(self.param.T_hat)

        self.depthwise_conv = tf.keras.layers.Conv1D(filters=self.param.H,
                                                     kernel_size=self.param.P,
                                                     dilation_rate=self.dilation,
                                                     padding=self.causal,
                                                     use_bias=False)
        self.prelu2 = tf.keras.layers.PReLU()
        self.residual_conv = tf.keras.layers.Conv1D(filters=self.param.B,
                                                    use_bias=False)
        self.skipconn_conv = tf.keras.layers.Conv1D(filters=self.param.Sc,
                                                    use_bias=False)

    def call(self, block_inputs):
        """
        Args:
            block_inputs: (, T_hat, B)

        Locals:
            bottelneck_outputs: (, T_hat, H)
            depthwise_inputs:  (, T_hat, H, 1)
            depthwise_outputs: (, T_hat, H)

        Returns:
            residual_outputs: (, T_hat, B)
            skipconn_outputs: (, T_hat, Sc)
        """
        # (, T_hat, B) -> (, T_hat, H)
        bottleneck_outputs = self.bottleneck_conv(block_inputs)
        # (, T_hat, H) -> (, T_hat, H)
        depthwise_inputs = self.prelu1(bottleneck_outputs)
        # (, T_hat, H) -> (, T_hat, H)
        depthwise_outputs = self.depthwise_conv(depthwise_inputs)
        # (, T_hat, H) -> (, T_hat, H)
        depthwise_outputs = self.prelu2(depthwise_outputs)
        # (, T_hat, H) -> (, T_hat, B)
        residual_outputs = self.residual_conv(depthwise_outputs)
        # (, T_hat, B), (, T_hat, B) -> (, T_hat, B)
        residual_outputs = residual_outputs + block_inputs
        # (, T_hat, H) -> (, T_hat, Sc)
        skipconn_outputs = self.skipconn_conv(depthwise_outputs)
        return residual_outputs, skipconn_outputs

    def get_config(self):
        return {**self.param.get_config(), "Dilation": self.dilation}
# Conv1DBlock end


class TemporalConvNet(tf.keras.layers.Layer):
    """Dilated Temporal Convolution Network (Dilated-TCN)

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        conv1dblock_list (List[Conv1DBlock]): List of the 1-D convolutional blocks
    """

    __slots__ = ("param", "conv1dblock_list")

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(TemporalConvNet, self).__init__(**kwargs)
        self.param = param
        self.conv1dblock_list = []
        for _ in range(self.param.R):
            for x in range(self.param.X):
                self.conv1dblock_list.append(Conv1DBlock(self.param, 2**x))

    def call(self, tcn_inputs):
        """
        Args:
            tcn_inputs: (, T_hat, B)

        Locals:
            residual_outputs: (, T_hat, B)
            skipconn_outputs: (, T_hat, Sc)

        Returns:
            tcn_outputs: (, T_hat, Sc)
        """
        # (, T_hat, Sc)
        tcn_outputs = tf.zeros(shape=(self.param.T_hat, self.param.Sc))
        for block in self.conv1dblock_list:
            # (, T_hat, B) -> (, T_hat, B), (, T_hat, Sc)
            residual_outputs, skipconn_outputs = block(tcn_inputs)
            # (, T_hat, B) -> (, T_hat, B)
            tcn_inputs = residual_outputs
            # (, T_hat, Sc) -> (, T_hat, Sc)
            tcn_outputs += skipconn_outputs
        return tcn_outputs

    def get_config(self):
        return self.param.get_config()
# TemporalConvNet end


class GlobalLayerNorm(tf.keras.layers.Layer):
    """Global Layer Normalization (i.e., gLN)

    Attributes:
        gamma (tf.Variable): Trainable parameter of shape=(, 1, num_features)
        beta (tf.Varaible): Trainable paramter of shape=(, 1, num_features)
        epsilon (float): Small constant for numerical stability
    """

    __slots__ = ("gamma", "beta", "epsilon")

    def __init__(self, num_features: int, eps: float = 1e-8, **kwargs):
        super(GlobalLayerNorm, self).__init__(**kwargs)
        self.epsilon = eps
        gamma_init = tf.random_normal_initializer()
        self.gamma = tf.Variable(
            initial_value=gamma_init(shape=(1, num_features)), trainable=True)
        beta_init = tf.zeros_initializer()
        self.beta = tf.Variable(
            initial_value=beta_init(shape=(1, num_features)), trainable=True)

    def call(self, inputs):
        """
        Args:
            inputs: (, T_hat, num_features)

        Locals:
            mean: (, 1, 1)
            var: (, 1, 1)

        Returns:
            outputs: (, T_hat, num_features)
        """
        # mean, var = tf.nn.moments()
        # mean, var = tf.nn.moments()
        pass
# GlobalLayerNorm end


class CumulativeLayerNorm(tf.keras.layers.Layer):
    """Cumulative Layer Normalization (i.e., cLN)

    Attributes:
        gamma (tf.Variable): Trainable parameter of shape=(, 1, num_features)
        beta (tf.Varaible): Trainable paramter of shape=(, 1, num_features)
        epsilon (float): Small constant for numerical stability
    """

    __slots__ = ("gamma", "beta", "epsilon")

    def __init__(self, num_features: int, eps: float = 1e-8, **kwargs):
        super(CumulativeLayerNorm, self).__init__(**kwargs)
        self.epsilon = eps
        gamma_init = tf.random_normal_initializer()
        self.gamma = tf.Variable(
            initial_value=gamma_init(shape=(1, num_features)), trainable=True)
        beta_init = tf.zeros_initializer()
        self.beta = tf.Variable(
            initial_value=beta_init(shape=(1, num_features)), trainable=True)

    def call(self, inputs):
        """
        Args:
            inputs: (, k, num_features) where k <= T_hat

        Locals:
            mean: (, k, 1)
            var: (, k, 1)

        Returns:
            outputs: (, k, num_features)
        """
        # (, k, num_features) -> (, k, 1), (, k, 1)
        mean, var = tf.nn.moments(inputs, axes=-1, keepdims=True)
        # (, k, num_features), (, k, 1), (, k, 1), (, k, num_features) -> (, k, num_features)
        outputs = self.gamma * (inputs - mean) / \
            tf.math.sqrt(var + self.epsilon) + self.beta
        return outputs
# CumulativeLayerNorm end
