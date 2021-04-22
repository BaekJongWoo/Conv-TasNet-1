import tensorflow as tf

from .layer import Encoder, Separater, Decoder
from .loss import SDR
from .param import ConvTasNetParam


class ConvTasNet(tf.keras.Model):

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNet, self).__init__(**kwargs)
        self.param = param
        self.encoder = Encoder(param)
        self.separater = Separater(param)
        self.decoder = Decoder(param)

        self.reshape = tf.keras.layers.Reshape(
            target_shape=[param.That, param.C, param.N])
        self.permute = tf.keras.layers.Permute([2, 1, 3])
        self.apply_mask = tf.keras.layers.Multiply()

    def call(self, mixture_segments):
        # (, That, L) -> (, That, N)
        mixture_weights = self.encoder(mixture_segments)

        # (, That, N) -> (, C, That, N)
        source_masks = self.separater(mixture_weights)

        # (, That, N) -> (, That, C*N)
        mixture_weights = tf.keras.layers.concatenate(
            [mixture_weights for _ in range(self.param.C)], axis=-1)

        # (, That, C*N) -> (, That, C, N)
        mixture_weights = self.reshape(mixture_weights)

        # (, That, C, N) -> (, C, That, N)
        mixture_weights = self.permute(mixture_weights)

        # (, C, That, N) -> (, C, That, N)
        source_weights = self.apply_mask([mixture_weights, source_masks])

        # (, C, That, N) -> (, C, That, L)
        estimated_sources = self.decoder(source_weights)

        return estimated_sources

    def get_config(self) -> dict:
        return self.param.get_config()

    @staticmethod
    def make(param: ConvTasNetParam,
             optimizer: tf.keras.optimizers.Optimizer = 'adam'):
        conv_tasnet = ConvTasNet(param)
        conv_tasnet.compile(optimizer=optimizer, loss=SDR())
        conv_tasnet.build(input_shape=(None, param.That, param.L))
        return conv_tasnet
