import tensorflow as tf
from tcn import TCN


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
    # T_hat | Total number of sample
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

    __slots__ = ('param', 'activation', 'do_gated_encoding'
                 'input_reshape', 'conv1d', 'output_reshape')

    # [param] activation: nonlinear function (optional) for the result of 1-D convolution.
    # [param] do_gated_encoding: gating mechanism handler (orginal model does not use!)
    def __init__(self, param: ConvTasNetParam, activation: str = "relu", do_gated_encoding: bool = False, **kwargs):
        super(ConvTasNetEncoder, self).__init__(**kwargs)
        self.param = param
        self.activation = activation
        self.do_gated_encoding = do_gated_encoding
        self.input_reshape = tf.keras.layers.Reshape(
            target_shape=(self.param.T_hat, self.param.L, 1))
        self.conv1d = tf.keras.layers.Conv2D(filters=self.param.N,
                                             kernel_size=(1, self.param.L),
                                             activation=self.activation,
                                             padding="valid")
        # TODO | implement gating mechanism
        self.output_reshape = tf.keras.layers.Reshape(
            target_shape=(self.param.T_hat, self.param.N))

    def call(self, encoder_inputs):
        reshaped_inputs = self.input_reshape(encoder_inputs)
        conv1d_outputs = self.conv1d(reshaped_inputs)  # main encoding process
        reshaped_outputs = self.output_reshape(conv1d_outputs)
        return reshaped_outputs

    def get_config(self):
        return {**self.param.get_config(),
                'Activation': self.activation,
                'Gating mechanism': self.do_gated_encoding}
# ConvTasNetEncoder end


class ConvTasNetSeparator(tf.keras.layers.Layer):

    """Separator using Dilated Temporal Convolutional Network (Dilated-TCN)"""

    __slots__ = ('param', 'is_causal', 'layer_normalization',
                 'input_conv1x1', 'tcn', 'prelu', 'output_conv1x1')

    def __init__(self, param: ConvTasNetParam, is_causal: bool = True, **kwargs):
        super(ConvTasNetSeparator, self).__init__(**kwargs)
        self.param = param
        self.is_causal = is_causal
        self.layer_normalization = None
        self.input_conv1x1 = None
        self.tcn = TCN(self.param)
        self.prelu = tf.keras.layers.PReLU()
        self.output_conv1x1 = None

    def call(self, separator_inputs):
        pass

    def get_config(self):
        return {**self.param.get_config(),
                'Causality': self.is_causal}
# ConvTasNetSeparator end


class ConvTasNetDecoder(tf.keras.layers.Layer):

    """Convolutional Decoder"""

    __slots__ = ('param')

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetDecoder, self).__init__(**kwargs)
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

    __slots__ = ('param', 'is_causal', 'encoder_activation', 'do_gated_encoding',
                 'encoder', 'separator', 'decoder')

    @staticmethod
    def make(param: ConvTasNetParam, optimizer: tf.keras.optimizers.Optimizer, loss: tf.keras.losses.Loss):
        model = ConvTasNet(param, )
        model.compile(optimizer=optimizer, loss=loss)
        model.build(input_shape=(None, param.T_hat, param.L))
        return model

    # [param] encoder_activation: nonlinear function (optional) for the convolutional encoding.
    # [param] do_gated_encoding: gating mechanism handler for the convolutional encoding (orginal model does not use!)
    def __init__(self, param: ConvTasNetParam, is_causal: bool = True, do_gated_encoding: bool = False, encoder_activation: str = 'relu', **kwargs):
        super(ConvTasNet, self).__init__(**kwargs)
        self.param = param
        self.is_causal = is_causal
        self.do_gated_encoding = do_gated_encoding
        self.encoder_activation = encoder_activation
        self.encoder = ConvTasNetEncoder(
            self.param, activation=self.encoder_activation, is_gated_encoding=self.do_gated_encoding)
        self.separator = ConvTasNetSeparator(self.param, self.is_causal)
        self.decoder = ConvTasNetDecoder(self.param)

    def call(self, inputs):
        # Encoding (1-D Convolution)
        # shape of encoder_output: T_hat x N
        encoder_outputs = self.encoder(inputs)
        # Separation (TCN)
        separator_outputs = self.separator(encoder_outputs)
        # Decoding (1-D Convolution)
        decoder_inputs = tf.keras.layers.Multiply()(encoder_outputs, separator_outputs)
        decoder_outputs = self.decoder(decoder_inputs)
        return decoder_outputs

    def get_config(self):
        return {**self.param.get_config(),
                'Causality': self.is_causal,
                'Encoder activation': self.encoder_activation,
                'Gated encoder': self.do_gated_encoding}
# ConvTasnet end
