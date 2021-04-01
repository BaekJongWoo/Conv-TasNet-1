"""
Tensorflow Implementation of the Fully-Convolutional Time-domain Audio Separation Network (Conv-TasNet)

Authors:
    kaparoo

References:
    [1] Luo Y., Mesgarani N. (2019). Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation,
            IEEE/ACM TRANSACTION ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, 27(8), 1256-1266,
            https://dl.acm.org/doi/abs/10.1109/TASLP.2019.2915167

    [2] https://github.com/naplab/Conv-TasNet
    [3] https://github.com/kaituoxu/Conv-TasNet
    [4] https://github.com/paxbun/TasNet
"""

import tensorflow as tf
from config import ConvTasNetParam
from tcn import TemporalConvNet


class ConvTasNetEncoder(tf.keras.layers.Layer):
    """1-D Convolutional Encoder

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        activation (str, optional): Nonlinear activation function for the conv1d_U layer
        reshape1 (keras.layers.Reshape): (, T_hat, L) -> (, T_hat, L, 1)
        conv1d_U (keras.layers.Conv2D): 1-D convolution layer to estimate weights of the mixture_segments
        gating (bool, optional): Flag for the gating mechanism
        conv1d_G (keras.layers.Conv2D): 1-D convolution layer corresponing to the gating mechanism
        multiply (keras.layers.Multiply): Elementwise muliplication layer
        reshape2 (keras.layers.Reshape): (, T_hat, N, 1) -> (, T_hat, N)
    """

    __slots__ = ("param", "activation", "reshape1", "conv1d_U",
                 "gating", "conv1d_G", "multiply", "reshape2")

    def __init__(self, param: ConvTasNetParam, activation: str = "relu", gating: bool = False, **kwargs):
        super(ConvTasNetEncoder, self).__init__(**kwargs)
        self.param = param
        self.activation = activation
        self.reshape1 = tf.keras.layers.Reshape(
            target_shape=(self.param.T_hat, self.param.L, 1))
        self.conv1d_U = tf.keras.layers.Conv2D(filters=self.param.N,
                                               kernel_size=(1, self.param.L),
                                               use_bias=False,
                                               activation=self.activation)
        self.gating = gating
        self.conv1d_G = tf.keras.layers.Conv2D(filters=self.param.N,
                                               kernel_size=(1, self.param.L),
                                               use_bias=False,
                                               activation="sigmoid")
        self.multiply = tf.keras.layers.Multiply()
        self.reshape2 = tf.keras.layers.Reshape(
            target_shape=(self.param.T_hat, self.param.N))

    def call(self, mixture_segments):
        """
        Args:
            mixture_segments: (, T_hat, L)

        Returns:
            mixture_weights: (, T_hat, N)
        """
        # (, T_hat, L) -> (, T_hat, L, 1)
        mixture_segments = self.reshape1(mixture_segments)
        # (, T_hat, L, 1) -> (, T_hat, N, 1)
        mixture_weights = self.conv1d_U(mixture_segments)
        # gating mechanism
        if self.gating:
            # (, T_hat, L, 1) -> (, T_hat, N, 1)
            gate_outputs = self.conv1d_G(mixture_segments)
            # (, T_hat, N, 1), (, T_hat, N, 1) -> (, T_hat, N, 1)
            mixture_weights = self.multiply([mixture_weights, gate_outputs])
        # (, T_hat, N, 1) -> (, T_hat, N)
        mixture_weights = self.reshape2(mixture_weights)
        return mixture_weights

    def get_config(self):
        return {**self.param.get_config(),
                "Activation": self.activation,
                "Gating mechanism": self.gating}
# ConvTasNetEncoder end


class ConvTasNetSeparator(tf.keras.layers.Layer):
    """Separator using Dilated Temporal Convolution Networks (i.e., Dilated-TCN)

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        layer_normalization (keras.layers.LayerNormalization): Normalization layer
        reshape1 (keras.layers.Reshape): (, T_hat, N) -> (, T_hat, N, 1)
        input_conv1x1 (keras.layers.Conv2D): 1x1 convolution layer
        reshape2 (keras.layers.Reshape): (, T_hat, B, 1) -> (, T_hat, B)
        TCN (TemporalConvNet): Dilated-TCN
        prelu (keras.layers.PReLU): Parametric recified linear unit (i.e., PReLU) activation layer
        reshape3 (keras.layers.Reshape): (, T_hat, Sc) -> (, T_hat, Sc, 1)
        output_conv1x1 (keras.layer.Conv2D): 1x1 convolution layer
        reshape3 (keras.layers.Reshape): (, T_hat, C*N) -> (, T_hat, C, N)
        softmax (keras.layers.Softmax): Softmax activation layer
    """

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetSeparator, self).__init__(**kwargs)
        self.param = param
        self.layer_normalization = tf.keras.layers.LayerNormalization(
            epsilon=self.param.eps)
        self.reshape1 = tf.keras.layers.Reshape(
            target_shape=(self.param.T_hat, self.param.N, 1))
        self.input_conv1x1 = tf.keras.layers.Conv2D(filters=self.param.B,
                                                    kernel_size=(
                                                        1, self.param.N),
                                                    use_bias=False)
        self.reshape2 = tf.keras.layers.Reshape(
            target_shape=(self.param.T_hat, self.param.B))
        self.TCN = TemporalConvNet(self.param)
        self.prelu = tf.keras.layers.PReLU()
        self.reshape3 = tf.keras.layers.Reshape(
            target_shape=(self.param.T_hat, self.param.Sc, 1))
        self.output_conv1x1 = tf.keras.layers.Conv2D(filters=self.param.C * self.param.N,
                                                     kernel_size=(
                                                         1, self.param.Sc),
                                                     use_bias=False)
        self.reshape4 = tf.keras.layers.Reshape(
            target_shape=(self.param.T_hat, self.param.C, self.param.N))
        self.softmax = tf.keras.layers.Softmax(
            axis=-2)  # normalization axis = sel.param.C

    def call(self, mixture_weights):
        """
        Args:
            mixture_weights: (, T_hat, N)

        Returns:
            estimated_masks: (, T_hat, C, N)
        """
        # (, T_hat, N) -> (, T_hat, N)
        mixture_weights = self.layer_normalization(mixture_weights)
        # (, T_hat, N) -> (, T_hat, N, 1)
        mixture_weights = self.reshape1(mixture_weights)
        # (, T_hat, N, 1) -> (, T_hat, B, 1)
        tcn_inputs = self.input_conv1x1(mixture_weights)
        # (, T_hat, B, 1) -> (, T_hat, B)
        tcn_inputs = self.reshape2(tcn_inputs)
        # (, T_hat, B) -> (, T_hat, Sc)
        tcn_outputs = self.TCN(tcn_inputs)
        # (, T_hat, Sc) -> (, T_hat, Sc)
        tcn_outputs = self.prelu(tcn_outputs)
        # (, T_hat, Sc) -> (, T_hat, Sc, 1)
        tcn_outputs = self.reshape3(tcn_outputs)
        # (, T_hat, Sc, 1) -> (, T_hat, C*N, 1)
        estimated_masks = self.output_conv1x1(tcn_outputs)
        # (, T_hat, C*N, 1) -> (, T_hat, C, N)
        estimated_masks = self.reshape4(estimated_masks)
        # (, T_hat, C, N) -> (, T_hat, C, N)
        estimated_masks = self.softmax(estimated_masks)
        return estimated_masks

    def get_config(self):
        return self.get_config()
# ConvTasNetSeparator end


class ConvTasNetDecoder(tf.keras.layers.Layer):
    """1-D Convolutional Decoder

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        multiply (keras.layers.Multiply): Elementwise multiplication layer
        reshape1 (keras.layers.Reshape): (, T_hat, C, N) -> (, T_hat, C, N, 1)
        conv1d_V (keras.layers.Conv2D): 1-D transpose convolution layer to estimate the sources of the original mixture
        reshape2 (keras.layers.Reshape): (, T_hat, C, L, 1) -> (, T_hat, C, L)
        permute (keras.layers.Permute): (, T_hat, C, L) -> (, C, T_hat, L)
    """

    __slots__ = ('param', 'multiply', 'reshape1',
                 'conv1d_V', 'reshape2', 'permute')

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetDecoder, self).__init__(**kwargs)
        self.param = param
        self.multiply = tf.keras.layers.Multiply()
        self.reshape1 = tf.keras.layers.Reshape(
            target_shape=(self.param.T_hat, self.param.C, self.param.N, 1))
        self.conv1d_V = tf.keras.layers.Conv2D(filters=self.param.L,
                                               kernel_size=(1, self.param.N),
                                               use_bias=False)
        self.reshape2 = tf.keras.layers.Reshape(
            target_shape=(self.param.T_hat, self.param.C, self.param.L))
        self.permute = tf.keras.layers.Permute((2, 1, 3))

    def call(self, mixture_weights, estimated_masks):
        """
        Args:
            mixture_weights: (, T_hat, N)
            estimated_masks: (, T_hat, C, N)

        Returns:
            estimated_sources: (, C, T_hat, L)
        """
        # (, T_hat, N) -> (, T_hat, 1, N)
        mixture_weights = tf.expand_dims(mixture_weights, 2)
        # (, T_hat, 1, N), (, T_hat, C, N) -> (, T_hat, C, N)
        estimated_weights = self.multiply([mixture_weights, estimated_masks])
        # (, T_hat, C, N) -> (, T_hat, C, N, 1)
        estimated_weights = self.reshape1(estimated_weights)
        # (, T_hat, C, N, 1) -> (, T_hat, C, L, 1)
        estimated_sources = self.conv1d_V(estimated_weights)
        # (, T_hat, C, L, 1) -> (, T_hat, C, L)
        estimated_sources = self.reshape2(estimated_sources)
        # (, T_hat, C, L) -> (, C, T_hat, L)
        estimated_sources = self.permute(estimated_sources)
        return estimated_sources

    def get_config(self):
        return self.param.get_config()
# ConvTasNetDecoder end


class ConvTasNet(tf.keras.Model):
    """Fully-Convolutional Time-domain Audio Separation Network (Conv-TasNet)

    Attributes:
        param (ConvTasNetParam): Hyperprameters
        gating (bool, optional): Flag for the gating mechanism (not in [1])
        encoder_activation (std, optional): Nonlinear activation function for the conv1d_U layer
        encoder (ConvTasNetEncoder): 1-D convolutional encoder
        separator (ConvTasNetSeparator): Dilated-TCN based Separator
        decoder (ConvTasNetDecoder): 1-D convolutional decoder
    """

    __slots__ = ('param', 'gating', 'encoder_activation',
                 'encoder', 'separator', 'decoder')

    @ staticmethod
    def make(param: ConvTasNetParam, optimizer: tf.keras.optimizers.Optimizer, loss: tf.keras.losses.Loss):
        model = ConvTasNet(param)
        model.compile(optimizer=optimizer, loss=loss)
        model.build(input_shape=(None, param.T_hat, param.L))
        return model

    def __init__(self, param: ConvTasNetParam, gating: bool = False, encoder_activation: str = "relu", **kwargs):
        super(ConvTasNet, self).__init__(**kwargs)
        self.param = param
        self.gating = gating
        self.encoder_activation = encoder_activation
        self.encoder = ConvTasNetEncoder(self.param, gating=self.gating)
        self.separator = ConvTasNetSeparator(self.param)
        self.decoder = ConvTasNetDecoder(self.param)

    def call(self, mixture_segments):
        """
        Args:
            mixture_segments: (, T_hat, L)

        Returns:
            estimated_sources: (, C, T_hat, L)
        """
        # (, T_hat, L) -> (, T_hat, N)
        mixture_weights = self.encoder(mixture_segments)
        # (, T_hat, N) -> (, T_hat, C, N)
        estimated_masks = self.separator(mixture_weights)
        # # (, T_hat, N), (, T_hat, C, N) -> (, C, T_hat, L)
        estimated_sources = self.decoder(mixture_weights, estimated_masks)
        return estimated_sources

    def get_config(self):
        return {**self.param.get_config(),
                'Encoder activation': self.encoder_activation,
                'Gating mechanism': self.gating}
# ConvTasnet end
