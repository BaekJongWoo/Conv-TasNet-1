import tensorflow as tf


class LayerNormInterface(tf.keras.layers.Layer):
    """Layer Normalization Interface

    Attributes:
        H (int): Number of features
        beta (tf.Varaible): Trainable paramter of shape=(, 1, H)
        gamma (tf.Variable): Trainable parameter of shape=(, 1, H)
        eps (float): Small constant for numerical stability
        multiply (keras.layers.Multiply): Elementwise multiplication layer
    """

    def __init__(self, H: int, eps: float = 1e-8, **kwargs):
        super(LayerNormInterface, self).__init__(**kwargs)
        self.H = H
        _gamma_init = tf.random_normal_initializer()
        self.gamma = tf.Variable(initial_value=_gamma_init(shape=(1, self.H)),
                                 trainable=True)
        _beta_init = tf.zeros_initializer()
        self.beta = tf.Variable(initial_value=_beta_init(shape=(1, self.H)),
                                trainable=True)
        self.eps = eps
        self.multiply = tf.keras.layers.Multiply()

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
        multiply (keras.layers.Multiply): Elementwise multiplication layer
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
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        outputs = self.multiply([self.gamma, inputs - mean]) / \
            tf.math.sqrt(var + self.eps) + self.beta
        return outputs
# GlobalLayerNorm end


class CumulativeLayerNorm(LayerNormInterface):
    """Cumulative Layer Normalization (i.e., cLN)

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
        for k in range(inputs.shape[-2]):
            sub_inputs = inputs[:, :k+1]
            sub_mean, sub_var = tf.nn.moments(
                sub_inputs, axes=[1, 2], keepdims=True)
            sub_outputs = self.multiply([self.gamma, inputs[:, k] - sub_mean]) / \
                tf.math.sqrt(sub_var + self.eps) + self.beta
            outputs.append(sub_outputs)
        outputs = tf.concat(outputs, -2)
        return outputs
# CumulativeLayerNorm end


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
