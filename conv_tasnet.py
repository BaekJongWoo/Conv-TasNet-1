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
        T (int): Original length of the mixture
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
        overlap (int): overlapping factor (e.g., overlap=2 means 50% (=1/2) overlap)
    """

    __slots__ = ('T', 'T_hat', 'C', 'N', 'L', 'B', 'Sc', 'H',
                 'P', 'X', 'R', 'causality', 'eps', 'overlap')

    def __init__(self, T: int, T_hat: int, C: int, N: int, L: int, B: int, Sc: int, H: int, P: int, X: int, R: int, causality: bool = True, eps: float = 1e-8, overlap: int = 2):
        self.T, self.T_hat = T, T_hat
        self.C, self.N, self.L = C, N, L
        self.B, self.Sc = B, Sc
        self.H, self.P = H, P
        self.X, self.R = X, R
        self.overlap, = overlap,
        self.causality = causality
        self.eps = eps  # for cLN, gLN (tcn.py)

    def get_config(self):
        return {'T': self.T, 'T_hat': self.T_hat,
                'C': self.C, 'N': self.N, 'L': self.L,
                'B': self.B, 'Sc': self.Sc,
                'H': self.H, 'P': self.P,
                'X': self.X, 'R': self.R,
                'causality': self.causality,
                'overlap': self.overlap,
                'eps': self.eps}
# ConvTasNetParam end


class ConvTasNetEncoder(tf.keras.layers.Layer):
    """1-D Convolutional Encoder

    Attributes:
        param: Hyperparameters
        activation (optional): Nonlinear function for the result of conv1d_U.
        conv1d_U: 1-D convolutional layer to estimate weights of the mixture_segments
        gating: Gating mechanism flag
        conv1d_G: 1-D convolutional layer for the gating mechanism (not in the paper[1])
        multiply: Layer for the elementwise muliplication
    """

    __slots__ = ('param', 'activation', 'conv1d_U',
                 'gating', 'conv1d_G', 'multiply')

    def __init__(self, param: ConvTasNetParam, activation: str = 'relu', gating: bool = False, **kwargs):
        super(ConvTasNetEncoder, self).__init__(**kwargs)
        self.param = param
        self.activation = activation
        self.conv1d_U = tf.keras.layers.Conv1D(filters=self.param.N,
                                               kernel_size=1,
                                               use_bias=False,
                                               padding='valid',
                                               activation=self.activation)
        self.gating = gating
        if(self.gating):
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
        # w = H(xU) where H is optional nonlinear activation function
        mixture_weights = self.conv1d_U(mixture_segments)
        # gating mechanism
        if(self.gating):
            # [T_hat x L] => [T_hat x N]
            gate_outputs = self.conv1d_G(mixture_segments)
            # [T_hat x N] * [T_hat x N] => [T_hat x N]
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
        param: Hyperparameters
        normalization: Causality depended layer normalization
        input_conv1x1: 1x1 convolution
        TCN: Temporal Convolution Network
        prelu: Paramertric recified linear unit (with training parameter 'alpha')
        output_conv1x1: 1x1 convolution (with sigmoid activation)
    """

    __slots__ = ('param' 'normalization',
                 'input_conv1x1', 'TCN', 'prelu', 'output_conv1x1')

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetSeparator, self).__init__(**kwargs)
        self.param = param
        self.input_conv1x1 = tf.keras.layers.Conv1D(
            filters=self.param.B, kernel_size=1, use_bias=False)
        self.TCN = TCN(self.param)
        self.prelu = tf.keras.layers.PReLU()
        # Unsure between Conv1D, and Conv1DTranspose
        self.output_conv1x1 = tf.keras.layers.Conv1D(filters=self.param.C * self.param.N,
                                                     kernel_size=1,
                                                     use_bias=False,
                                                     activation='sigmoid')
        self.output_reshape = tf.keras.layers.Reshape(
            target_shape=(self.param.T_hat, self.param.C, self.param.N))

    def call(self, mixture_weights):
        """
        Args:
            mixture_weights: [T_hat x N]

        Returns:
            estimated_masks: [T_hat x C x N]
        """
        # TODO | add causality-depended normalization for mixture_weights
        # [T_hat x N] => [T_hat x B]
        tcn_inputs = self.input_conv1x1(mixture_weights)
        # [T_hat x B] => [T_hat x B]
        tcn_outputs = self.TCN(tcn_inputs)
        # [T_hat x B] => [T_hat x B]
        tcn_outputs = self.prelu(tcn_outputs)
        # [T_hat x B] => [T_hat x CN] (with sigmoid activation)
        conv1x1_outputs = self.output_conv1x1(tcn_outputs)
        # [T_hat x CN] => [T_hat x C x N]
        estimated_masks = self.output_reshape(conv1x1_outputs)
        return estimated_masks

    def get_config(self):
        return self.param.get_config()
# ConvTasNetSeparator end


class ConvTasNetDecoder(tf.keras.layers.Layer):

    """1-D Convolutional Decoder"""

    __slots__ = ('param', 'multiply', 'transConv1d')

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetDecoder, self).__init__(**kwargs)
        self.param = param
        self.multiply = tf.keras.layers.Multiply()
        self.transConv1d = tf.keras.layers.Conv1D()  # must fix this line

    def concatenate(self, estimated_sources):
        pass

    def call(self, mixture_weights, estimated_masks):
        """
        Args:
            mixture_weights: [T_hat x N]
            estimated_masks: [T_hat x C x N]

        Locals:
            estimated_weights: [T_hat x C X N]
            estimated_sources: [T_hat x C x L]

        Returns:
            concatenated_sources: [C x T]
        """
        # mixture_weights: [T_hat x N] => [T_hat x 1 x N] (2 - not 1 - because we have to consider index of the batch_size: 0)
        mixture_weights = tf.expand_dims(mixture_weights, 2)
        estimated_weights = self.multiply(mixture_weights, estimated_masks)
        estimated_sources = self.transConv1d(estimated_weights)
        # TODO | must concatenate the estimated_sources to get estimated_sources with consider overlapping
        concatenated_sources = estimated_sources  # must fix this line
        return concatenated_sources

    def get_config(self):
        return self.param.get_config()
# ConvTasNetDecoder end


class ConvTasNet(tf.keras.Model):

    """Fully-Convolutional Time-domain Audio Separation Network

    Attributes:
        param: Hyperprameters
        gating: Gating mechanism flag (not in the paper[1])
        encoder: 1-D convolutional encoder
        separator: TCN based Separator
        decoder: 1-D convolutional decoder
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
            estimated_sources: [C x T]
        """
        # [T_hat x L] => [T_hat x N]
        mixture_weights = self.encoder(mixture_segments)
        # [T_hat x N] => [T_hat x C x N]
        estimated_masks = self.separator(mixture_weights)
        # [T_hat x N], [T_hat x C x N] => [C x T]
        estimated_sources = self.decoder(mixture_weights, estimated_masks)
        return estimated_sources

    def get_config(self):
        return {**self.param.get_config(),
                'Encoder activation': self.encoder_activation,
                'Gating mechanism': self.gating}
# ConvTasnet end
