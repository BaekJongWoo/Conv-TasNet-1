import tensorflow as tf


class SISNR(tf.keras.losses.Loss):

    """SI-SNR: Scale-Invariant Source-to-Noise Ratio"""

    __slots__ = 'eps'

    def __init__(self, eps: float = 1e-10, **kwargs):
        super(SISNR, self).__init__(**kwargs)
        self.eps = eps  # small constant for numerical stability

    def call(self, s, s_hat):
        s_target = (tf.reduce_sum(tf.multiply(s, s_hat)) / tf.reduce_sum(tf.multiply(s, s))) * s
	e_noise = s_hat - s_target
        return 20 * tf.math.log(tf.norm(s_target) / (tf.norm(e_noise) + self.eps) + self.eps)


class SDR(tf.keras.losses.Loss):

    """SDR: Source-to-Distortion Ratio"""

    __slots__ = 'eps'  # small constant for numerical stability

    def __init__(self,  eps: float = 1e-10, **kwargs):
        super(SDR, self).__init__(**kwargs)
        self.eps = eps

    def call(self, s, s_hat):
        return 20 * tf.math.log(tf.norm(s_hat - s) / (tf.norm(s) + self.eps) + self.eps)
