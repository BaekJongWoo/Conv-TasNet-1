import tensorflow as tf


class SISNR(tf.keras.losses.Loss):
    """
    SI-SNR: Scale-Invariant Source-to-Noise Ratio

    Attributes:
        eps (float): small constant for numerical stability
    """

    __slots__ = 'eps'

    def __init__(self, eps: float = 1e-8, **kwargs):
        super(SISNR, self).__init__(**kwargs)
        self.eps = eps

    def call(self, s, s_hat) -> float:
        s_target = (tf.reduce_sum(tf.multiply(s, s_hat)) /
                    tf.reduce_sum(tf.multiply(s, s))) * s
        e_noise = s_hat - s_target
        return 20 * tf.math.log(tf.norm(s_target) / (tf.norm(e_noise) + self.eps) + self.eps)
# SISNR end


class SDR(tf.keras.losses.Loss):
    """
    SDR: Source-to-Distortion Ratio

    Attributes:
        eps (float): small constant for numerical stability
    """

    __slots__ = 'eps'

    def __init__(self,  eps: float = 1e-8, **kwargs):
        super(SDR, self).__init__(**kwargs)
        self.eps = eps

    def call(self, s, s_hat) -> float:
        return 20 * tf.math.log(tf.norm(s_hat - s) / (tf.norm(s) + self.eps) + self.eps)
# SDR end
