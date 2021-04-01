"""
Tensorflow 2.0 Implementation of the Fully-Convolutional Time-domain Audio Separation Network (Conv-TasNet)

Authors:
    kaparoo

References:
    [1] Y. Luo and N. Mesgarani, "Conv-TasNet: Surpassing Ideal Time?Frequency Magnitude Masking for Speech Separation,"
        in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 27, no. 8, pp. 1256-1266, Aug. 2019,
        doi: 10.1109/TASLP.2019.2915167.

    [2] https://github.com/naplab/Conv-TasNet
    [3] https://github.com/kaituoxu/Conv-TasNet
    [4] https://github.com/paxbun/TasNet
"""

import tensorflow as tf
from config import ConvTasNetParam
from tcn import TemporalConvNet


class ConvTasNetEncoder(tf.keras.layers.Layer):
    """Encoding Module using 1-D Convolution

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        gating (bool, optional): Flag of gating mechanism
        activation (str, optional): Nonlinear activation function for conv1d_U layer
        reshape1 (keras.layers.Reshape): (, T_hat, L) -> (, T_hat, L, 1)
        conv1d_U (keras.layers.Conv2D): 1-D convolution layer estimating weights of mixture segments
        conv1d_G (keras.layers.Conv2D): 1-D convolution layer corresponding to the gating mechanism
        multiply (keras.layers.Multiply): Layer for elementwise muliplication
        reshape2 (keras.layers.Reshape): (, T_hat, N, 1) -> (, T_hat, N)
    """

    __slots__ = ("param", "activation", "reshape1", "conv1d_U",
                 "gating", "conv1d_G", "multiply", "reshape2")

    def __init__(self, param: ConvTasNetParam, gating: bool = False, activation: str = "relu", **kwargs):
        super(ConvTasNetEncoder, self).__init__(**kwargs)
        self.param = param
        self.gating = gating
        self.activation = activation

        self.reshape1 = tf.keras.layers.Reshape(
            target_shape=(self.param.T_hat, self.param.L, 1))
        self.conv1d_U = tf.keras.layers.Conv2D(filters=self.param.N,
                                               kernel_size=(1, self.param.L),
                                               use_bias=False,
                                               activation=self.activation)
        self.conv1d_G = tf.keras.layers.Conv2D(filters=self.param.N,
                                               kernel_size=(1, self.param.L),
                                               use_bias=False,
                                               activation="sigmoid")
        self.multiply = tf.keras.layers.Multiply()
        self.reshape2 = tf.keras.layers.Reshape(
            target_shape=(self.param.T_hat, self.param.N))

        # self.conv1d_U = tf.keras.layers.Conv1D(filters=self.param.N,
        #                                        kernel_size=1,
        #                                        use_bias=False,
        #                                        activation=self.activation)
        # self.conv1d_G = tf.keras.layers.Conv1D(filters=self.param.N,
        #                                        kernel_size=1,
        #                                        use_bias=False,
        #                                        activation="sigmoid")
        # self.multiply = tf.keras.layers.Multiply()

    def call(self, mixture_segments):
        """
        Args:
            mixture_segments: (, T_hat, L)

        Locals:
            gate_outputs: (, T_hat, N)

        Returns:
            mixture_weights: (, T_hat, N)
        """
        mixture_segments = self.reshape1(mixture_segments)
        mixture_weights = self.conv1d_U(mixture_segments)
        if self.gating:  # gating mechanism
            gate_outputs = self.conv1d_G(mixture_segments)
            mixture_weights = self.multiply([mixture_weights, gate_outputs])
        mixture_weights = self.reshape2(mixture_weights)

        # mixture_weights = self.conv1d_U(mixture_segments)
        # if self.gating:  # gating mechanism
        #     gate_outputs = self.conv1d_G(mixture_segments)
        #     mixture_weights = self.multiply([mixture_weights, gate_outputs])

        return mixture_weights

    def get_config(self):
        return {**self.param.get_config(),
                "Gating mechanism": self.gating,
                "Encoder activation": self.activation}
# ConvTasNetEncoder end


class ConvTasNetSeparator(tf.keras.layers.Layer):
    """Separation Module using Dilated Temporal Convolution Networks (i.e., Dilated-TCN)

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        layer_normalization (keras.layers.LayerNormalization): Normalization layer
        reshape1 (keras.layers.Reshape): (, T_hat, N) -> (, T_hat, N, 1)
        input_conv1x1 (keras.layers.Conv2D): 1x1 convolution layer (bottleneck)
        reshape2 (keras.layers.Reshape): (, T_hat, B, 1) -> (, T_hat, B)
        TCN (TemporalConvNet): Dilated-TCN
        prelu (keras.layers.PReLU): Layer for parametric recified linear unit (i.e., PReLU) activation
        reshape3 (keras.layers.Reshape): (, T_hat, Sc) -> (, T_hat, Sc, 1)
        output_conv1x1 (keras.layer.Conv2D): 1x1 convolution layer (reverse bottleneck)
        reshape3 (keras.layers.Reshape): (, T_hat, C*N) -> (, T_hat, C, N)
        softmax (keras.layers.Softmax): Softmax activation layer
    """

    # __slots__ = ("param", "layer_normalization", "input_conv1x1",
    #             "TCN", "prelu", "output_conv1x1")

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
        self.softmax = tf.keras.layers.Softmax(axis=-2)

        # self.input_conv1x1 = tf.keras.layers.Conv1D(filters=self.param.B,
        #                                             kernel_size=1,
        #                                             use_bias=False)
        # self.TCN = TemporalConvNet(self.param)
        # self.prelu = tf.keras.layers.PReLU()
        # self.output_conv1x1 = tf.keras.layers.Conv1D(filters=self.param.C * self.param.N,
        #                                              kernel_size=1,
        #                                              use_bias=False)
        # self.output_reshape = tf.keras.layers.Reshape(
        #     target_shape=(self.param.T_hat, self.param.C, self.param.N))
        # self.softmax = tf.keras.layers.Softmax(axis=-2)

    def call(self, mixture_weights):
        """
        Args:
            mixture_weights: (, T_hat, N)

        Locals:
            tcn_inputs: (, T_hat, B)
            tcn_outputs: (, T_hat, Sc)

        Returns:
            estimated_masks: (, T_hat, C, N)
        """
        mixture_weights = self.layer_normalization(mixture_weights)
        mixture_weights = self.reshape1(mixture_weights)
        tcn_inputs = self.input_conv1x1(mixture_weights)
        tcn_inputs = self.reshape2(tcn_inputs)
        tcn_outputs = self.TCN(tcn_inputs)
        tcn_outputs = self.prelu(tcn_outputs)
        tcn_outputs = self.reshape3(tcn_outputs)
        estimated_masks = self.output_conv1x1(tcn_outputs)
        estimated_masks = self.reshape4(estimated_masks)
        estimated_masks = self.softmax(estimated_masks)

        # mixture_weights = self.layer_normalization(mixture_weights)
        # tcn_inputs = self.input_conv1x1(mixture_weights)
        # tcn_outputs = self.TCN(tcn_inputs)
        # tcn_outputs = self.prelu(tcn_outputs)
        # estimated_masks = self.output_conv1x1(tcn_outputs)
        # estimated_masks = self.output_reshape(estimated_masks)
        # estimated_masks = self.softmax(estimated_masks)

        return estimated_masks

    def get_config(self):
        return self.get_config()
# ConvTasNetSeparator end


class ConvTasNetDecoder(tf.keras.layers.Layer):
    """Decoding Module using 1-D Convolution

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        multiply (keras.layers.Multiply): Layer for elementwise multiplication
        reshape1 (keras.layers.Reshape): (, T_hat, C, N) -> (, T_hat, C, N, 1)
        conv1d_V (keras.layers.Conv2D): 1-D transpose convolution layer estimating sources of the original mixture segments
        reshape2 (keras.layers.Reshape): (, T_hat, C, L, 1) -> (, T_hat, C, L)
        permute (keras.layers.Permute): (, T_hat, C, L) -> (, C, T_hat, L)
    """

    __slots__ = ("param", "multiply", "reshape1",
                 "conv1d_V", "reshape2", "permute")

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

        # self.multiply = tf.keras.layers.Multiply()
        # self.conv1d_V = tf.keras.layers.Conv1D(filters=self.param.L,
        #                                        kernel_size=1,
        #                                        use_bias=False)
        # self.permute = tf.keras.layers.Permute((2, 1, 3))

    def call(self, mixture_weights, estimated_masks):
        """
        Args:
            mixture_weights: (, T_hat, N)
            estimated_masks: (, T_hat, C, N)

        Locals:
            source_weights: (, T_hat, C, N)

        Returns:
            estimated_sources: (, C, T_hat, L)
        """
        mixture_weights = tf.expand_dims(mixture_weights, 2)
        source_weights = self.multiply([mixture_weights, estimated_masks])
        source_weights = self.reshape1(source_weights)
        estimated_sources = self.conv1d_V(source_weights)
        estimated_sources = self.reshape2(estimated_sources)
        estimated_sources = self.permute(estimated_sources)

        # mixture_weights = tf.expand_dims(mixture_weights, 2)
        # source_weights = self.multiply([mixture_weights, estimated_masks])
        # estimated_sources = self.conv1d_V(source_weights)
        # estimated_sources = self.permute(estimated_sources)

        return estimated_sources

    def get_config(self):
        return self.param.get_config()
# ConvTasNetDecoder end


class ConvTasNet(tf.keras.Model):
    """Fully-Convolutional Time-domain Audio Separation Network (Conv-TasNet)

    Attributes:
        param (ConvTasNetParam): Hyperprameters
        encoder_gating (bool, optional): Flag for gating mechanism (not in [1])
        encoder_activation (str, optional): Nonlinear activation function for the encoder
        encoder (ConvTasNetEncoder): Encoding module using 1-D convolution
        separator (ConvTasNetSeparator): Separation module using Dilated-TCN
        decoder (ConvTasNetDecoder): Decoding module using 1-D convolution
    """

    __slots__ = ("param", "encoder_gating", "encoder_activation",
                 "encoder", "separator", "decoder")

    @ staticmethod
    def make(param: ConvTasNetParam, optimizer: tf.keras.optimizers.Optimizer, loss: tf.keras.losses.Loss):
        model = ConvTasNet(param)
        model.compile(optimizer=optimizer, loss=loss)
        model.build(input_shape=(None, param.T_hat, param.L))
        return model

    def __init__(self, param: ConvTasNetParam, encoder_gating: bool = False, encoder_activation: str = "relu", **kwargs):
        super(ConvTasNet, self).__init__(**kwargs)
        self.param = param
        self.encoder_gating = encoder_gating
        self.encoder_activation = encoder_activation
        self.encoder = ConvTasNetEncoder(self.param,
                                         gating=self.encoder_gating,
                                         activation=self.encoder_activation)
        self.separator = ConvTasNetSeparator(self.param)
        self.decoder = ConvTasNetDecoder(self.param)

    def call(self, mixture_segments):
        """
        Args:
            mixture_segments: (, T_hat, L)

        Locals:
            mixture_weights: (, T_hat, N)
            estimated_masks: (, T_hat, C, N)

        Returns:
            estimated_sources: (, C, T_hat, L)
        """
        mixture_weights = self.encoder(mixture_segments)
        estimated_masks = self.separator(mixture_weights)
        estimated_sources = self.decoder(mixture_weights, estimated_masks)
        return estimated_sources

    def get_config(self):
        return {**self.param.get_config(),
                "Gating mechanism": self.encoder_gating,
                "Encoder activation": self.encoder_activation}
# ConvTasnet end
