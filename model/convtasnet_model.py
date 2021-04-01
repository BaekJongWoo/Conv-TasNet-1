"""
Tensorflow 2.0 Implementation of the Fully-Convolutional Time-domain Audio Separation Network (Conv-TasNet)

Authors:
    kaparoo

References:
    [1] Y. Luo and N. Mesgarani, "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation,"
        in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 27, no. 8, pp. 1256-1266, Aug. 2019,
        doi: 10.1109/TASLP.2019.2915167.
    [2] https://github.com/naplab/Conv-TasNet
    [3] https://github.com/kaituoxu/Conv-TasNet
    [4] https://github.com/paxbun/TasNet
"""

import tensorflow as tf
from convtasnet_param import ConvTasNetParam
from temporalconvnet import TemporalConvNet


class ConvTasNetEncoder(tf.keras.layers.Layer):
    """Encoding Module using 1-D Convolution

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        conv1d_U (keras.layers.Conv1D): 1-D convolution layer estimating weights of mixture segments
        conv1d_G (keras.layers.Conv1D): 1-D convolution layer corresponding to the gating mechanism
        multiply (keras.layers.Multiply): Layer for elementwise muliplication
    """

    __slots__ = ("param", "conv1d_U", "conv1d_G", "multiply")

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetEncoder, self).__init__(**kwargs)
        self.param = param
        self.conv1d_U = tf.keras.layers.Conv1D(filters=self.param.N,
                                               activation="linear",
                                               use_bias=False)
        self.conv1d_G = tf.keras.layers.Conv1D(filters=self.param.N,
                                               activation="sigmoid",
                                               use_bias=False)
        self.multiply = tf.keras.layers.Multiply()

    def call(self, mixture_segments):
        """
        Args:
            mixture_segments: (, T_hat, L)

        Locals:
            gate_outputs: (, T_hat, N)

        Returns:
            mixture_weights: (, T_hat, N)
        """
        # (, T_hat, L) -> (, T_hat, N)
        mixture_weights = self.conv1d_U(mixture_segments)
        if self.param.gating:  # gating mechanism
            # (, T_hat, L) -> (, T_hat, N)
            gate_outputs = self.conv1d_G(mixture_segments)
            # (, T_hat, N), (, T_hat, N) -> (, T_hat, N)
            mixture_weights = self.multiply([mixture_weights, gate_outputs])
        return mixture_weights

    def get_config(self):
        return self.param.get_config()
# ConvTasNetEncoder end


class ConvTasNetSeparator(tf.keras.layers.Layer):
    """Separation Module using Dilated Temporal Convolution Network (i.e., Dilated-TCN)

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        layer_normalization (keras.layers.LayerNormalization): Normalization layer
        input_conv1x1 (keras.layers.Conv1D): 1x1 convolution layer
        TCN (TemporalConvNet): Dilated-TCN layer
        prelu (keras.layers.PReLU): PReLU activation layer
        output_conv1x1 (keras.layers.Conv1D): 1x1 convolution layer
        output_reshape (keras.layers.Reshape): (, T_hat, C*N) -> (, T_hat, C, N)
        softmax (keras.layers.Softmax): Softmax activation layer
    """

    __slots__ = ("param", "layer_normalization", "input_conv1x1", "TCN",
                 "prelu", "output_conv1x1", "outpus_reshape", "softmax")

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetSeparator, self).__init__(**kwargs)
        self.param = param
        self.layer_normalization = tf.keras.layers.LayerNormalization()
        self.input_conv1x1 = tf.keras.layers.Conv1D(filters=self.param.B,
                                                    use_bias=False)
        self.TCN = TemporalConvNet(self.param)
        self.prelu = tf.keras.layers.PReLU()
        self.output_conv1x1 = tf.keras.layers.Conv1D(filters=self.param.C * self.param.N,
                                                     use_bias=False)
        self.output_reshape = tf.keras.layers.Reshape(
            target_shape=(self.param.T_hat, self.param.C, self.param.N))
        self.softmax = tf.keras.layers.Softmax(axis=-2)

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
        # (, T_hat, N) -> (, T_hat, N)
        mixture_weights = self.layer_normalization(mixture_weights)
        # (, T_hat, N) -> (, T_hat, B)
        tcn_inputs = self.input_conv1x1(mixture_weights)
        # (, T_hat, B) -> (, T_hat, Sc)
        tcn_outputs = self.TCN(tcn_inputs)
        # (, T_hat, Sc) -> (, T_hat, Sc)
        tcn_outputs = self.prelu(tcn_outputs)
        # (, T_hat, Sc) -> (, T_hat, C*N)
        estimated_masks = self.output_conv1x1(tcn_outputs)
        # (, T_hat, C*N) -> (, T_hat, C, N)
        estimated_masks = self.output_reshape(estimated_masks)
        # (, T_hat, C, N) -> (, T_hat, C, N)
        estimated_masks = self.softmax(estimated_masks)
        return estimated_masks

    def get_config(self):
        return self.get_config()
# ConvTasNetSeparator end


class ConvTasNetDecoder(tf.keras.layers.Layer):
    """Decoding Module using 1-D Convolution

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        multiply (keras.layers.Multiply): Layer for elementwise multiplication
        conv1d_V (keras.layers.Conv1D): 1-D transpose convolution layer estimating
                                        sources of the original mixture segments
        permute (keras.layers.Permute): (, T_hat, C, L) -> (, C, T_hat, L)
    """

    __slots__ = ("param", "multiply", "conv1d_V", "permute")

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetDecoder, self).__init__(**kwargs)
        self.param = param
        self.multiply = tf.keras.layers.Multiply()
        self.conv1d_V = tf.keras.layers.Conv1D(filters=self.param.L,
                                               use_bias=False)
        self.permute = tf.keras.layers.Permute((2, 1, 3))

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
        # (, T_hat, N) -> (, T_hat, 1, N)
        mixture_weights = tf.expand_dims(mixture_weights, 2)
        # (, T_hat, 1, N), (, T_hat, C, N) -> (, T_hat, C, N)
        source_weights = self.multiply([mixture_weights, estimated_masks])
        # (, T_hat, C, N) -> (, T_hat, C, L)
        estimated_sources = self.conv1d_V(source_weights)
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
        encoder (ConvTasNetEncoder): Encoding module using 1-D convolution
        separator (ConvTasNetSeparator): Separation module using Dilated-TCN
        decoder (ConvTasNetDecoder): Decoding module using 1-D convolution
    """

    __slots__ = ("param", "encoder", "separator", "decoder")

    @ staticmethod
    def make(param: ConvTasNetParam, optimizer: tf.keras.optimizers.Optimizer, loss: tf.keras.losses.Loss):
        model = ConvTasNet(param)
        model.compile(optimizer=optimizer, loss=loss)
        model.build(input_shape=(None, param.T_hat, param.L))
        return model

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNet, self).__init__(**kwargs)
        self.param = param
        self.encoder = ConvTasNetEncoder(self.param)
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
        # (, T_hat, L) -> (, T_hat, N)
        mixture_weights = self.encoder(mixture_segments)
        # (, T_hat, N) -> (, T_hat, C, N)
        estimated_masks = self.separator(mixture_weights)
        # (, T_hat, N), (, T_hat, C, N) -> (, C, T_hat, L)
        estimated_sources = self.decoder(mixture_weights, estimated_masks)
        return estimated_sources

    def get_config(self):
        return self.param.get_config()
# ConvTasnet end
