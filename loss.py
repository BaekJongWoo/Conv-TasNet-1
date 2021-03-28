import tensorflow as tf


class SISNR(tf.keras.losses.Loss):

    """SI-SNR: Scale-Invariant Signal-to-Noise Ratio"""

    __slots__ = 'eps'

    def __init__(self, eps: float = 1e-10, **kwargs):
        super(SISNR, self).__init__(**kwargs)
        self.eps = eps

    def call(self, s, s_hat):
        s_target = (tf.matmul(tf.transpose(s), s_hat) / (tf.norm(s)**2)) * s
        e_noise = s_hat - s_target
        return 20 * tf.math.log(tf.norm(s_target) / (tf.norm(e_noise) + self.eps) + self.eps)


class SDR(tf.keras.losses.Loss):

    """SDR: Signal-to-Distortion Ratio"""

    __slots__ = 'eps'

    def __init__(self,  eps: float = 1e-10, **kwargs):
        super(SDR, self).__init__(**kwargs)
        self.eps = eps

    def call(self, s, s_hat):
        return 20 * tf.math.log(tf.norm(s_hat - s) / (tf.norm(s) + self.eps) + self.eps)
