import tensorflow as tf


class SISNR(tf.keras.losses.Loss):
    """Scale-invariant Source-to-Noise Ratio Loss (i.e., inverse of SI-SNR).

    Attributes:
        eps (float): A small constant for numerical stability
    """

    def __init__(self, eps: float = 1e-8, **kwargs):
        super(SISNR, self).__init__(**kwargs)
        self.eps = eps

    def call(self, s: tf.Tensor, s_hat: tf.Tensor):
        s_target = (tf.reduce_sum(tf.multiply(s, s_hat)) /
                    tf.reduce_sum(tf.multiply(s, s))) * s
        e_noise = s_hat - s_target
        loss = 20 * tf.math.log(tf.norm(e_noise) /
                                (tf.norm(s_target) + self.eps) + self.eps) / tf.math.log(10.0)
        return loss
# SISNR end


class SDR(tf.keras.losses.Loss):
    """Source-to-Distortion Ratio Loss (i.e, inverse of SDR)

    Attributes:
        eps (float): small constant for numerical stability
    """

    def __init__(self, eps: float = 1e-8, **kwargs):
        super(SDR, self).__init__(**kwargs)
        self.eps = eps

    def call(self, s: tf.Tensor, s_hat: tf.Tensor):
        loss = 20 * tf.math.log(tf.norm(s - s_hat) /
                                (tf.norm(s) + self.eps) + self.eps) / tf.math.log(10.0)
        return loss
# SDR end
