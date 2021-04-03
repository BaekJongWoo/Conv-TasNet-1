import tensorflow as tf


class GlobalLayerNorm(tf.keras.layers.Layer):
    """Global Layer Normalization (i.e., gLN)

    Attributes:
        H (int): Number of features
        gamma (tf.Variable): Trainable parameter of shape=(, 1, H)
        beta (tf.Varaible): Trainable paramter of shape=(, 1, H)
        eps (float): Small constant for numerical stability
        multiply (keras.layers.Multiply): Elementwise multiplication layer
    """

    def __init__(self, H: int, eps: float = 1e-8, **kwargs):
        super(GlobalLayerNorm, self).__init__(**kwargs)
        self.H = H
        _gamma_init = tf.random_normal_initializer()
        self.gamma = tf.Variable(initial_value=_gamma_init(shape=(1, self.H)),
                                 trainable=True)
        _beta_init = tf.zeros_initializer()
        self.beta = tf.Variable(initial_value=_beta_init(shape=(1, self.H)),
                                trainable=True)
        self.eps = eps
        self.multiply = tf.keras.layers.Multiply()

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


class CumulativeLayerNorm(tf.keras.layers.Layer):
    """Cumulative Layer Normalization (i.e., cLN)

    Attributes:
        H (int): Number of features
        gamma (tf.Variable): Trainable parameter of shape=(, 1, H)
        beta (tf.Varaible): Trainable paramter of shape=(, 1, H)
        eps (float): Small constant for numerical stability
        multiply (keras.layers.Multiply): Elementwise multiplication layer
    """

    def __init__(self, H: int, eps: float = 1e-8, **kwargs):
        super(CumulativeLayerNorm, self).__init__(**kwargs)
        self.H = H
        _gamma_init = tf.random_normal_initializer()
        self.gamma = tf.Variable(initial_value=_gamma_init(shape=(1, self.H)),
                                 trainable=True)
        _beta_init = tf.zeros_initializer()
        self.beta = tf.Variable(initial_value=_beta_init(shape=(1, self.H)),
                                trainable=True)
        self.eps = eps
        self.multiply = tf.keras.layers.Multiply()

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
