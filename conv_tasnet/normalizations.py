import tensorflow as tf


class LayerNormInterface(tf.keras.layers.Layer):
    """Custom Layer Normalization Interface.

    Attributes:
        prefix (str): Name of the normalization among 'gLN', 'cLN', and 'eLN'
        beta (tf.Varaible): Trainable paramter of shape=(, 1, N)
        gamma (tf.Variable): Trainable parameter of shape=(, 1, N)
        eps (float): Small constant for numerical stability
    """

    def __init__(self, prefix: str, N: int, eps: float = 1e-8, **kwargs):
        super(LayerNormInterface, self).__init__(**kwargs)
        # for any custom trainable parameter,
        # must set its name to save its weight!
        self.gamma = self.add_weight(name=f"{prefix}_gamma",
                                     shape=(1, N),
                                     initializer="random_normal",
                                     trainable=True)
        self.beta = self.add_weight(name=f"{prefix}_beta",
                                    shape=(1, N),
                                    initializer="zeros",
                                    trainable=True)
        self.eps = eps

    def call(self, input: tf.Tensor) -> tf.Tensor:
        """
        Args:
            inputs (tf.Tensor): Tensor of shape=(, K, N)

        Returns:
            outputs (tf.Tensor): Tensor of shape=(, K, N)
        """
        raise NotImplementedError(
            "`call` function of the layer normalization must be implemented!")
# LayerNormInterface end


class GlobalLayerNorm(LayerNormInterface):
    """Global Layer Normalization (i.e., gLN).

    Description:
        Layer normalization for `noncausal` system
        Inherited from LayerNormInterface

    Attributes:
        beta (tf.Varaible): Trainable paramter of shape=(, 1, N)
        gamma (tf.Variable): Trainable parameter of shape=(, 1, N)
        eps (float): Small constant for numerical stability
    """

    def __init__(self, N: int, eps: float = 1e-8, **kwargs):
        super(GlobalLayerNorm, self).__init__("gLN", N=N, eps=eps, **kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Args:
            inputs (tf.Tensor): Tensor of shape=(, K, N)

        Returns:
            outputs (tf.Tensor): Tensor of shape=(, K, N)
        """
        _mean = tf.reduce_mean(inputs,
                               axis=[-2, -1],
                               keepdims=True)
        _var = tf.reduce_mean(tf.pow(inputs - _mean, 2),
                              axis=[-2, -1],
                              keepdims=True)
        outputs = self.gamma * ((inputs - _mean) /
                                tf.sqrt(_var + self.eps)) + self.beta
        return outputs
# GlobalLayerNorm end


class CausalLayerNorm(LayerNormInterface):
    """Cumulative Layer Normalization (i.e., cLN).

    Description:
        Layer normalization for `causal` system
        Inherited from LayerNormInterface

    Attributes:
        beta (tf.Varaible): Trainable paramter of shape=(, 1, N)
        gamma (tf.Variable): Trainable parameter of shape=(, 1, N)
        eps (float): Small constant for numerical stability
    """

    def __init__(self, N: int, eps: float = 1e-8, **kwargs):
        super(CausalLayerNorm, self).__init__("cLN", N=N, eps=eps, **kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Args:
            inputs (tf.Tensor): Tensor of shape=(, K, N)

        Returns:
            outputs (tf.Tensor): Tensor of shape=(, K, N)
        """
        _K = inputs.shape[-2]
        _count = tf.reshape(range(1, _K+1), [1, _K, 1])
        _count = tf.cast(_count, dtype=tf.float32)

        _H_mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        _H_pow_mean = tf.reduce_mean(tf.pow(inputs, 2), axis=-1, keepdims=True)

        _sum = tf.cumsum(_H_mean, axis=-2)
        _pow_sum = tf.cumsum(_H_pow_mean, axis=-2)

        _mean = _sum / _count
        _var = (_pow_sum - 2*_mean*_sum) / _count + tf.pow(_mean, 2)

        outputs = self.gamma * ((inputs - _mean) /
                                tf.sqrt(_var + self.eps)) + self.beta
        return outputs
# CausalLayerNorm end


class ExponentLayerNorm(LayerNormInterface):
    """Exponential Layer Normalization (i.e., eLN).

    Description:
        Layer normalization for `causal` system
        Inherited from LayerNormInterface

    Attributes:
        alpha (float): Forgetting rate
        beta (tf.Varaible): Trainable paramter of shape=(, 1, N)
        gamma (tf.Variable): Trainable parameter of shape=(, 1, N)
        eps (float): Small constant for numerical stability
        omega (float): Exponent
    """

    def __init__(self, N, alpha: float = 0.5, eps: float = 1e-8, omega: float = 0.5, **kwargs):
        super(ExponentLayerNorm, self).__init__("eLN", N=N, eps=eps, **kwargs)
        self.alpha, self.omega = alpha, omega

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Args:
            inputs (tf.Tensor): Tensor of shape=(, K, N)
        Returns:
            outputs (tf.Tensor): Tensor of shape=(, K, N)
        """
        return inputs  # TODO | must fix this line
# ExponentialLayerNorm end
