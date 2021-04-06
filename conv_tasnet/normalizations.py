import tensorflow as tf


class LayerNormInterface(tf.keras.layers.Layer):
    """Layer Normalization Interface

    Attributes:
        H (int): Number of features
        beta (tf.Varaible): Trainable paramter of shape=(, 1, H)
        gamma (tf.Variable): Trainable parameter of shape=(, 1, H)
        eps (float): Small constant for numerical stability
    """

    def __init__(self, H: int, eps: float = 1e-8, **kwargs):
        super(LayerNormInterface, self).__init__(**kwargs)
        self.H = H
        self.gamma = self.add_weight(shape=(1, self.H),
                                     initializer="random_normal",
                                     trainable=True)
        self.beta = self.add_weight(shape=(1, self.H),
                                    initializer="zeros",
                                    trainable=True)
        self.eps = eps

    def call(self, input: tf.Tensor) -> tf.Tensor:
        """
        Args:
            inputs (tf.Tensor): Tensor of shape=(, K, H)

        Returns:
            outputs (tf.Tensor): Tensor of shape=(, K, H)
        """
        raise NotImplementedError("`call` function must be implemented!")
# LayerNormInterface end


class GlobalLayerNorm(LayerNormInterface):
    """Global Layer Normalization (i.e., gLN)

    Description:
        Layer normalization for `noncausal` system
        Inherited from LayerNormInterface

    Attributes:
        H (int): Number of features
        beta (tf.Varaible): Trainable paramter of shape=(, 1, H)
        gamma (tf.Variable): Trainable parameter of shape=(, 1, H)
        eps (float): Small constant for numerical stability
    """

    def __init__(self, H: int, eps: float = 1e-8, **kwargs):
        super(GlobalLayerNorm, self).__init__(H=H, eps=eps, **kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Args:
            inputs (tf.Tensor): Tensor of shape=(, K, H)

        Returns:
            outputs (tf.Tensor): Tensor of shape=(, K, H)
        """
        mean = tf.reshape(tf.reduce_mean(inputs, axis=[1, 2]), [-1, 1, 1])
        var = tf.reshape(tf.reduce_mean(
            (inputs-mean)**2, axis=[1, 2]), [-1, 1, 1])
        outputs = self.gamma * ((inputs - mean) /
                                (var + self.eps) ** 0.5) + self.beta
        return outputs
# GlobalLayerNorm end


class ChannelwiseLayerNorm(LayerNormInterface):
    """Channelwise Layer Normalization (i.e., cwLN)

    Description:
        Layer normalization for `causal` system
        Inherited from LayerNormInterface

    Attributes:
        H (int): Number of features
        beta (tf.Varaible): Trainable paramter of shape=(, 1, H)
        gamma (tf.Variable): Trainable parameter of shape=(, 1, H)
        eps (float): Small constant for numerical stability
    """

    def __init__(self, H: int, eps: float = 1e-8, **kwargs):
        super(ChannelwiseLayerNorm, self).__init__(H=H, eps=eps, **kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Args:
            inputs (tf.Tensor): Tensor of shape=(, K, H)

        Returns:
            outputs (tf.Tensor): Tensor of shape=(, K, H)
        """
        mean = tf.reshape(tf.reduce_mean(inputs, axis=[2]), [-1, 1, self.H])
        var = tf.reshape(tf.reduce_mean(
            (inputs-mean)**2, axis=[2]), [-1, 1, self.H])
        outputs = self.gamma * ((inputs - mean) /
                                (var + self.eps) ** 0.5) + self.beta
        return outputs
# ChannelwiseLayerNorm end


class CumulativeLayerNorm(LayerNormInterface):
    """Channelwise Layer Normalization (i.e., cwLN)

    Description:
        Layer normalization for `causal` system
        Inherited from LayerNormInterface

    Attributes:
        H (int): Number of features
        beta (tf.Varaible): Trainable paramter of shape=(, 1, H)
        gamma (tf.Variable): Trainable parameter of shape=(, 1, H)
        eps (float): Small constant for numerical stability
        multiply (keras.layers.Multiply): Elementwise multiplication layer
    """

    def __init__(self, H: int, eps: float = 1e-8, **kwargs):
        super(CumulativeLayerNorm, self).__init__(H=H, eps=eps, **kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Args:
            inputs (tf.Tensor): Tensor of shape=(, K, H)

        Returns:
            outputs (tf.Tensor): Tensor of shape=(, K, H)
        """
        outputs = []
        for k in range(inputs.shape[2]):
            sub_inputs = inputs[:, :k+1]
            sub_mean = tf.reshape(tf.reduce_mean(
                inputs, axis=[1, 2]), [-1, 1, 1])
            sub_var = tf.reshape(tf.reduce_mean(
                (sub_inputs - sub_mean)**2, axis=[1, 2]), [-1, 1, 1])
            sub_outputs = self.gamma * ((sub_inputs[:, k] - mean) /
                                        (var + self.eps) ** 0.5) + self.beta
            outputs.append(sub_outputs)
        outputs = tf.concat(outputs, -2)
        return outputs
# CumumlativeLayerNorm end


class ExponentialLayerNorm(LayerNormInterface):
    """Exponential Layer Normalization (i.e., eLN)

    Description:
        Layer normalization for `causal` system
        Inherited from LayerNormInterface

    Attributes:
        H (int): Number of features
        alpha (float): Forgetting rate
        beta (tf.Varaible): Trainable paramter of shape=(, 1, H)
        gamma (tf.Variable): Trainable parameter of shape=(, 1, H)
        eps (float): Small constant for numerical stability
        omega (float): Exponent
        multiply (keras.layers.Multiply): Elementwise multiplication layer
    """

    def __init__(self, H, alpha: float = 0.5, eps: float = 1e-8, omega: float = 0.5, **kwargs):
        super(ExponentialLayerNorm, self).__init__(H=H, eps=eps, **kwargs)
        self.alpha, self.omega = alpha, omega

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Args:
            inputs (tf.Tensor): Tensor of shape=(, K, H)
        Returns:
            outputs (tf.Tensor): Tensor of shape=(, K, H)
        """
        return inputs  # TODO | must fix this line
# ExponentialLayerNorm end
