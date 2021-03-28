import tensorflow as tf


class ConvTasNetParam:
    # ===============================================================================
    # Hyperparameters Description
    # ===============================================================================
    # N     | Number of filters in autoencoder
    # L     | Length of the filters (in sample)
    # B     | Number of channels in bottleneck and the residual paths' 1x1-conv blocks
    # Sc    | Number of channels in skip-connection paths' 1x1-conv blocks
    # H     | Number of channels in convolutional blocks
    # P     | Kernal size in convolutional blocks
    # X     | Number of convolutional blocks in each repeat
    # R     | Number of repeats
    # ===============================================================================
    # T-hat | Total number of sample
    # C     | Total number of source (i.e., class)
    # ===============================================================================

    # Reference
    # Luo Y., Mesgarani N. (2019). Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation,
    #     IEEE/ACM TRANSACTION ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, 27(8), 1256-1266, https://dl.acm.org/doi/abs/10.1109/TASLP.2019.2915167

    """Hyperparameters of the Conv-TasNet"""

    __slots__ = ('T_hat', 'C', 'N', 'L', 'B', 'Sc', 'H', 'P', 'X', 'R')

    def __init__(self, T_hat: int, C: int, N: int, L: int, B: int, Sc: int, H: int, P: int, X: int, R: int):
        self.T_hat, self.C = T_hat, C
        # Filter
        self.N, self.L = N, L
        # Convolutional block
        self.B, self.Sc = B, Sc  # 1x1 conv block
        self.H, self.P = H, P  # convolutional block
        self.X, self.R = X, R  # convolutional block repeat

    def get_config(self):
        return {
            'T_hat': self.T_hat, 'C': self.C,
            'N': self.N, 'L': self.L,
            'B': self.B, 'Sc': self.Sc,
            'H': self.H, 'P': self.P,
            'X': self.X, 'R': self.R
        }
# ConvTasNetParam end


class ConvTasNetEncoder(tf.keras.layers.Layer):

    """Convolution Encoder"""

    __slots__ = ('param', 'input_reshape', 'conv1d', 'output_reshape')

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetEncoder, self).__init__()
        self.param = param
        self.input_reshape = tf.keras.layers.Reshape((param.T, param.L, 1))
        self.conv1d = tf.keras.layers.Conv2D(filters=self.param.N,
                                             kernel_size=(1, self.param.L),
                                             activation="relu",
                                             padding="valid")
        self.output_reshape = tf.keras.layers.Reshape((param.T, param.N))

    def call(self, encoder_inputs):
        reshaped_inputs = self.input_reshape(encoder_inputs)
        conv1d_outputs = self.conv1d(reshaped_inputs)  # main encoding process
        reshaped_outputs = self.output_reshape(conv1d_outputs)
        return reshaped_outputs

    def get_config(self):
        return self.param.get_config()
# ConvTasNetEncoder end


class ConvTasNetSeparator(tf.keras.layers.Layer):

    """Separator using Dilated Temporal Convolutional Network (Dilated-TCN)"""

    __slots__ = ('param')

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetSeparator, self).__init__()
        self.param = param

    def call(self, separator_inputs):
        pass

    def get_config(self):
        return self.param.get_config()
# ConvTasNetSeparator end


class ConvTasNetDecoder(tf.keras.layers.Layer):

    """Convolutional Decoder"""

    __slots__ = ('param')

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetDecoder, self).__init__()
        self.param = param

    def call(self, decoder_inputs):
        pass

    def get_config(self):
        return self.param.get_config()
# ConvTasNetDecoder end


class ConvTasNet(tf.keras.Model):
    # References
    # https://github.com/naplab/Conv-TasNet
    # https://github.com/paxbun/TasNet

    """Conv-TasNet Implementation"""

    __slots__ = ('param', 'encoder', 'separator', 'decoder')

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
        # Encoding (1-D Convolution)
        encoder_outputs = self.encoder(inputs)
        # Separation (TCN)
        separator_outputs = self.separator(encoder_outputs)
        # Decoding (1-D Convolution)
        decoder_inputs = tf.keras.layers.Multiply()(
            encoder_outputs, separator_outputs)  # Multiply(*): elementwise multiplication
        decoder_outputs = self.decoder(decoder_inputs)
        return decoder_outputs

    def get_config(self):
        return self.param.get_config()
# ConvTasnet end
