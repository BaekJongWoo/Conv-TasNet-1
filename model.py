import tensorflow as tf


class ConvTasNetParam:
    # ===============================================================================
    # Hyperparameters Description
    # ===============================================================================
    # N  | Number of filters in autoencoder
    # L  | Length of the filters (in sample)
    # B  | Number of channels in bottleneck and the residual paths' 1x1-conv blocks
    # Sc | Number of channels in skip-connection paths' 1x1-conv blocks
    # H  | Number of channels in convolutional blocks
    # P  | Kernal size in convolutional blocks
    # X  | Number of convolutional blocks in each repeat
    # R  | Number of repeats
    # ===============================================================================

    # Reference
    # Luo Y., Mesgarani N. (2019). Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation,
    #     IEEE/ACM TRANSACTION ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, 27(8), 1256-1266, https://dl.acm.org/doi/abs/10.1109/TASLP.2019.2915167

    def __init__(self, T: int, N: int, L: int, B: int, Sc: int, H: int, P: int, X: int, R: int):
        # number of sample
        # TODO | unsure param.T(custom-added hyperparameter) is necessary
        self.T = T
        # filter
        self.N, self.L = N, L
        # convolutional block
        self.B, self.Sc = B, Sc  # 1x1 conv block
        self.H, self.P = H, P  # convolutional block
        self.X, self.R = X, R  # convolutional block repeat

    def get_config(self):
        return {
            # 'T': self.T,
            'N': self.N, 'L': self.L,
            'B': self.B, 'Sc': self.Sc,
            'H': self.H, 'P': self.P,
            'X': self.X, 'R': self.R
        }
# ConvTasNetParam end


class ConvTasNetEncoder(tf.keras.layers.Layer):
    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetEncoder, self).__init__()
        self.param = param

    def call(self, encoder_inputs):
        return

    def get_config(self):
        return self.param.get_config()
# ConvTasNetEncoder end


class ConvTasNetSeparator(tf.keras.layers.Layer):
    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetSeparator, self).__init__()
        self.param = param

    def call(self, separator_inputs):
        return

    def get_config(self):
        return self.param.get_config()
# ConvTasNetSeparator end


class ConvTasNetDecoder(tf.keras.layers.Layer):
    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetDecoder, self).__init__()
        self.param = param
        self.makeDecoderInputs = tf.keras.layers.Multiply()

    def call(self, decoder_inputs):
        return

    def get_config(self):
        return self.param.get_config()
# ConvTasNetDecoder end


class ConvTasNet(tf.keras.Model):
    # References
    # https://github.com/naplab/Conv-TasNet
    # https://github.com/paxbun/TasNet

    @staticmethod
    def make(param: ConvTasNetParam, optimizer: tf.keras.optimizers.Optimizer, loss: tf.keras.losses.Loss):
        model = ConvTasNet(param, )
        model.compile(optimizer=optimizer, loss=loss)
        # TODO | unsure param.T(custom-added hyperparameter) is necessary
        model.build(input_shape=(None, param.T, param.L))
        return model

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNet, self).__init__()
        self.param = param
        self.encoder = ConvTasNetEncoder(self.param)
        self.separator = ConvTasNetSeparator(self.param)
        self.decoder = ConvTasNetDecoder(self.param)

    def call(self, inputs):
        encoder_outputs = self.encoder(inputs)
        separator_outputs = self.separator(encoder_outputs)
        decoder_inputs = tf.keras.layers.Multiply()(
            encoder_outputs, separator_outputs)  # elementwise multiplication
        decoder_outputs = self.decoder(decoder_inputs)
        return decoder_outputs
# ConvTasnet end
