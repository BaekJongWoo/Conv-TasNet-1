"""
Tensorflow Implementation of the Fully-Convolutional Time-domain Audio Separation Network

Author: kaparoo

References:
    [1] Luo Y., Mesgarani N. (2019). Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation,
            IEEE/ACM TRANSACTION ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, 27(8), 1256-1266,
            https://dl.acm.org/doi/abs/10.1109/TASLP.2019.2915167

    [2] https://github.com/naplab/Conv-TasNet
    [3] https://github.com/kaituoxu/Conv-TasNet
    [4] https://github.com/paxbun/TasNet
"""

import tensorflow as tf
from tcn import TCN, cLN, gLN


class ConvTasNetParam:
    """
    Attributes:
        T_hat (int): Total number of sample
        C (int): Total number of source (i.e., class)
        N (int): Number of filters in autoencoder
        L (int): Length of the filters (in sample)
        B (int): Number of channels in bottleneck and the residual paths' 1x1-conv blocks
        Sc (int): Number of channels in skip-connection paths' 1x1-conv blocks
        H (int): Number of channels in convolutional blocks
        P (int): Kernal size in convolutional blocks
        X (int): Number of convolutional blocks in each repeat
        R (int): Number of repeats
        causality (bool): causality of the model
        eps (float): small constant for numerical stability
    """

    __slots__ = ('T_hat', 'C', 'N', 'L', 'B', 'Sc', 'H',
                 'P', 'X', 'R', 'causality', 'eps')

    def __init__(self, T_hat: int, C: int, N: int, L: int, B: int, Sc: int, H: int, P: int, X: int, R: int, causality: bool = True, eps: float = 1e-8):
        self.T_hat, self.C = T_hat, C
        self.N, self.L = N, L
        self.B, self.Sc = B, Sc
        self.H, self.P = H, P
        self.X, self.R = X, R
        self.causality = causality
        self.eps = eps  # for cLN, gLN (tcn.py)

    def get_config(self):
        return {'T_hat': self.T_hat, 'C': self.C,
                'N': self.N, 'L': self.L,
                'B': self.B, 'Sc': self.Sc,
                'H': self.H, 'P': self.P,
                'X': self.X, 'R': self.R,
                'causality': self.causality,
                'eps': self.eps}
# ConvTasNetParam end


class ConvTasNetEncoder(tf.keras.layers.Layer):
    """1-D Convolutional Encoder

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        activation (str, optional): Nonlinear activation function for the result of conv1d_U.
        conv1d_U (keras.layers.Conv1D): 1-D convolutional layer to estimate weights of the mixture_segments
        gating (bool, optional): Gating mechanism flag
        conv1d_G (keras.layers.Conv1D): 1-D convolutional layer for the gating mechanism (not in the paper[1])
        multiply (keras.layers.Multiply): Layer for the elementwise muliplication
    """

    __slots__ = ('param', 'activation', 'conv1d_U',
                 'gating', 'conv1d_G', 'multiply')

    def __init__(self, param: ConvTasNetParam, activation: str = 'relu', gating: bool = False, **kwargs):
        super(ConvTasNetEncoder, self).__init__(**kwargs)
        self.param = param
        self.activation = activation if activation == 'relu' else None
        self.conv1d_U = tf.keras.layers.Conv1D(filters=self.param.N,
                                               kernel_size=1,
                                               use_bias=False,
                                               padding='valid',
                                               activation=self.activation)
        self.gating = gating
        if self.gating:
            self.conv1d_G = tf.keras.layers.Conv1D(filters=self.param.N,
                                                   kernel_size=1,
                                                   use_bias=False,
                                                   padding='valid',
                                                   activation='sigmoid')
            self.multiply = tf.keras.layers.Multiply()

    def call(self, mixture_segments):
        """
        Args:
            mixture_segments: [T_hat x L]

        Returns:
            mixture_weights: [T_hat x N]
        """
        # w = H(xU) where H is optional nonlinear activation function (=ReLU)
        mixture_weights = self.conv1d_U(mixture_segments)
        # gating mechanism
        if self.gating:
            # [T_hat x L] -> [T_hat x N]
            gate_outputs = self.conv1d_G(mixture_segments)
            # [T_hat x N], [T_hat x N] -> [T_hat x N]
            mixture_weights = self.multiply(mixture_weights * gate_outputs)
        return mixture_weights

    def get_config(self):
        return {**self.param.get_config(),
                'Activation': self.activation,
                'Gating mechanism': self.gating}
# ConvTasNetEncoder end


class ConvTasNetSeparator(tf.keras.layers.Layer):
    """Separator using Dilated Temporal Convolutional Network (Dilated-TCN)

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        normalization (gLN): Global layer normalization
        input_conv1x1 (keras.layers.Conv1D): 1x1 convolution
        TCN (TCN): Dilated temporal convolution network
        prelu (keras.layers.PReLU): Paramertric recified linear unit
        output_conv1x1 (keras.layers.Conv1D): 1x1 convolution (with sigmoid activation)
        softmax (keras.layers.Softmax): Softmax activation for the unit summation constraint
    """

    __slots__ = ('param' 'normalization', 'input_conv1x1', 'TCN',
                 'prelu', 'output_conv1x1', 'output_reshape', 'softmax')

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetSeparator, self).__init__(**kwargs)
        self.param = param
        self.normalization = gLN(eps=self.param.eps)
        self.input_conv1x1 = tf.keras.layers.Conv1D(filters=self.param.B,
                                                    kernel_size=1,
                                                    use_bias=False)
        self.TCN = TCN(self.param)
        self.prelu = tf.keras.layers.PReLU()
        self.output_conv1x1 = tf.keras.layers.Conv1D(filters=self.param.C * self.param.N,
                                                     kernel_size=1,
                                                     use_bias=False,
                                                     activation='sigmoid')
        self.output_reshape = tf.keras.layers.Reshape(
            target_shape=(self.param.T_hat, self.param.C, self.param.N))
        self.softmax = tf.keras.layers.Softmax(axis=-2)  # axis: self.param.C

    def call(self, mixture_weights):
        """
        Args:
            mixture_weights: [T_hat x N]

        Returns:
            estimated_masks: [T_hat x C x N]
        """
        # [T_hat x N] -> [T_hat x N]
        mixture_weights = self.normalization(mixture_weights)
        # [T_hat x N] -> [T_hat x B]
        tcn_inputs = self.input_conv1x1(mixture_weights)
        # [T_hat x B] -> [T_hat x B]
        tcn_outputs = self.TCN(tcn_inputs)
        # [T_hat x B] -> [T_hat x B]
        tcn_outputs = self.prelu(tcn_outputs)
        # [T_hat x B] -> [T_hat x CN] (with sigmoid activation)
        conv1x1_outputs = self.output_conv1x1(tcn_outputs)
        # [T_hat x CN] -> [T_hat x C x N]
        estimated_masks = self.output_reshape(conv1x1_outputs)
        # [T_hat x CN] -> [T_hat x C x N]
        estimated_masks = self.softmax(estimated_masks)
        return estimated_masks

    def get_config(self):
        return self.param.get_config()
# ConvTasNetSeparator end


class ConvTasNetDecoder(tf.keras.layers.Layer):
    """1-D Convolutional Decoder

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        multiply (keras.layers.Multiply): elementwise multiplication
        trans_conv1d (keras.layers):
    """

    __slots__ = ('param', 'multiply', 'trans_conv1d')

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetDecoder, self).__init__(**kwargs)
        self.param = param
        self.multiply = tf.keras.layers.Multiply()
        self.trans_conv1d = tf.keras.layers.Conv1DTranspose(filters=self.param.L,
                                                            kernel_size=1, use_bias=False)

    def call(self, mixture_weights, estimated_masks):
        """
        Args:
            mixture_weights: [T_hat x N]
            estimated_masks: [T_hat x C x N]

        Returns:
            concatenated_sources: [C x T_hat x L]
        """
        # [T_hat x N] -> [T_hat x 1 x N]
        mixture_weights = tf.expand_dims(mixture_weights, 2)
        # [T_hat x 1 x N], [T_hat x C x N] -> [T_hat x C x N]
        estimated_weights = self.multiply(mixture_weights, estimated_masks)
        # [T_hat x C x N] -> [T_hat x C x L]
        estimated_sources = self.trans_conv1d(estimated_weights)
        return estimated_sources

    def get_config(self):
        return self.param.get_config()
# ConvTasNetDecoder end


class ConvTasNet(tf.keras.Model):
    """Fully-Convolutional Time-domain Audio Separation Network

    Attributes:
        param (ConvTasNetParam): Hyperprameters
        gating (bool, optional): Gating mechanism flag (not in the paper[1])
        encoder_activation (str, optional): nonlinear activation function
        encoder (ConvTasNetEncoder): 1-D convolutional encoder
        separator (ConvTasNetSeparator): TCN based Separator
        decoder (ConvTasNetDecoder): 1-D convolutional decoder
    """

    __slots__ = ('param', 'gating', 'encoder_activation',
                 'encoder', 'separator', 'decoder')

    @staticmethod
    def make(param: ConvTasNetParam, optimizer: tf.keras.optimizers.Optimizer, loss: tf.keras.losses.Loss):
        model = ConvTasNet(param)
        model.compile(optimizer=optimizer, loss=loss)
        model.build(input_shape=(None, param.T_hat, param.L))
        return model

    def __init__(self, param: ConvTasNetParam, gating: bool = False, encoder_activation: str = 'relu', **kwargs):
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
            mixture_segments: [T_hat x L]

        Returns:
            estimated_sources: [C x T_hat x L]
        """
        # [T_hat x L] => [T_hat x N]
        mixture_weights = self.encoder(mixture_segments)
        # [T_hat x N] => [T_hat x C x N]
        estimated_masks = self.separator(mixture_weights)
        # [T_hat x N], [T_hat x C x N] => [C x T_hat x L]
        estimated_sources = self.decoder(mixture_weights, estimated_masks)
        return estimated_sources

    def get_config(self):
        return {**self.param.get_config(),
                'Encoder activation': self.encoder_activation,
                'Gating mechanism': self.gating}
# ConvTasnet end
