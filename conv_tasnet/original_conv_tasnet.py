import tensorflow as tf
from .conv_tasnet_param import ConvTasNetParam
from .temporal_conv_net import TemporalConvNet
from .normalizations import GlobalLayerNorm as gLN
from .normalizations import CausalLayerNorm as cLN


class ConvTasNet(tf.keras.Model):
    """Conv-TasNet Model.

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        encoder (ConvTasNetEncoder): Encoding module using 1-D convolution
        decoder (ConvTasNetDecoder): Decoding module using 1-D convolution
        separator (ConvTasNetSeparator): Separation module using Dilated-TCN
    """

    @staticmethod
    def make(param: ConvTasNetParam, optimizer: tf.keras.optimizers.Optimizer, loss: tf.keras.losses.Loss):
        model = ConvTasNet(param)
        model.compile(optimizer=optimizer, loss=loss)
        model.build(input_shape=(None, param.K, param.L))
        return model

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNet, self).__init__(**kwargs)
        self.param = param
        self.encoder = ConvTasNetEncoder(self.param)
        self.separator = ConvTasNetSeparator(self.param)
        self.decoder = ConvTasNetDecoder(self.param)

    def call(self, mixture_segments: tf.Tensor) -> tf.Tensor:
        """
        Args:
            mixture_segments (tf.Tensor): Tensor of shape=(, K, L)

        Returns:
            estimated_sources (tf.Tensor): Tensor of shape=(, C, K, L) 
        """
        # (, K, L) -> (, K, N)
        mixture_weights = self.encoder(mixture_segments)
        # (, K, N) -> (, K, C, N)
        estimated_masks = self.separator(mixture_weights)
        # (, K, N), (, K, C, N) -> (, C, K, L)
        estimated_sources = self.decoder(mixture_weights, estimated_masks)
        return estimated_sources

    def get_config(self) -> dict:
        return self.param.get_config()
# ConvTasNet end


class ConvTasNetEncoder(tf.keras.layers.Layer):
    """Encoding Module using 1-D Convolution.

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        conv1d_U (tf.keras.layers.Conv1D): 1-D convolution layer for tne encoding
        conv1d_G (tf.keras.layers.Conv1D; optional): 1-D convolution layer for the gating machanism
        multiply (tf.keras.layers.Multiply; optional): Elementwise multiplication layer
    """

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetEncoder, self).__init__(**kwargs)
        self.param = param
        self.conv1d_U = tf.keras.layers.Conv1D(filters=self.param.N,
                                               kernel_size=1,
                                               activation="linear",
                                               use_bias=self.param.use_bias)
        if self.param.gating:
            self.conv1d_G = tf.keras.layers.Conv1D(filters=self.param.N,
                                                   kernel_size=1,
                                                   activation="sigmoid",
                                                   use_bias=self.param.use_bias)
            self.multiply = tf.keras.layers.Multiply()

    def call(self, mixture_segments: tf.Tensor) -> tf.Tensor:
        """
        Args:
            mixture_segments (tf.Tensor): Tensor of shape=(, K, L)

        Returns:
            mixture_weights (tf.Tensor): Tensor of shape=(, K, N) 
        """
        # (, K, L) -> (, K, N)
        mixture_weights = self.conv1d_U(mixture_segments)
        if self.param.gating:  # gating mechanism
            # (, K, L) -> (, K, N)
            gating_factor = self.conv1d_G(mixture_segments)
            # (, K, N), (, K, N) -> (, K, N)
            mixture_weights = self.multiply([mixture_weights, gating_factor])
        return mixture_weights

    def get_config(self) -> dict:
        return self.param.get_config()
# ConvTasNetEncoder end


class ConvTasNetDecoder(tf.keras.layers.Layer):
    """Decoding Module using 1-D Convolution.

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        multiply (tf.keras.layers.Multiply): Elementwise multiplication layer
        conv1d_V (tf.keras.layers.Conv1D): 1-D convolution layer
        permute (tf.keras.layers.Premute): Layer that changes index orders of tf.Tensor
    """

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetDecoder, self).__init__(**kwargs)
        self.param = param
        self.multiply = tf.keras.layers.Multiply()
        self.conv1_V = tf.keras.layers.Conv1D(filters=self.param.L,
                                              kernel_size=1,
                                              activation="linear",
                                              use_bias=self.param.use_bias)
        self.permute = tf.keras.layers.Permute((2, 1, 3))

    def call(self, mixture_weights: tf.Tensor, estimated_masks: tf.Tensor) -> tf.Tensor:
        """
        Args:
            mixture_weights (tf.Tensor): Tensor of shape=(, K, N)
            estimated_masks (tf.Tensor): Tensor of shape=(, K, C, N)

        Returns:
            estimated_sources (tf.Tensor): Tensor of shape=(, C, K, L)
        """
        # (, K, N) -> (, K, 1, N)
        mixture_weights = tf.expand_dims(mixture_weights, 2)
        # (, K, 1, N), (, K, C, N) -> (, K, C, N)
        sources_weights = self.multiply([mixture_weights, estimated_masks])
        # (, K, C, N) -> (, K, C, L)
        estimated_sources = self.conv1_V(sources_weights)
        # (, K, C, L) -> (, C, K, L)
        estimated_sources = self.permute(estimated_sources)
        return estimated_sources

    def get_config(self) -> dict:
        return self.param.get_config()
# ConvTasNetDecoder end


class ConvTasNetSeparator(tf.keras.layers.Layer):
    """Separation Module using Dilated Temporal Convolution Network (Dilated-TCN).

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        normalization (cLN | gLN): Normalization layer
        input_conv1x1 (tf.keras.layers.Conv1D): 1x1 convolution layer
        temporal_conv_net (TemporalConvNet): Dilated-TCN layer
        output_block (SeparatorOutputBlock): Output processing layer after the Dilated-TCN
    """

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetSeparator, self).__init__(**kwargs)
        self.param = param

        if self.param.causal:  # causal system
            self.normalization = cLN(N=self.param.N, eps=self.param.eps)
        else:  # noncausal system
            self.normalization = gLN(N=self.param.N, eps=self.param.eps)

        self.input_conv1x1 = tf.keras.layers.Conv1D(filters=self.param.B,
                                                    kernel_size=1,
                                                    activation="linear",
                                                    use_bias=self.param.use_bias)
        self.temporal_conv_net = TemporalConvNet(self.param)
        self.output_block = SeperatorOutputBlock(self.param)

    def call(self, mixture_weights: tf.Tensor) -> tf.Tensor:
        """
        Args:
            mixture_weights (tf.Tensor): Tensor of shape=(, K, N)

        Returns:
            esimated_masks (tf.Tensor): Tensor of shape=(, K, C, N)
        """
        # (, K, N) -> (, K, N)
        mixture_weights = self.normalization(mixture_weights)
        # (, K, N) -> (, K, B)
        tcn_inputs = self.input_conv1x1(mixture_weights)
        # (, K, B) -> (, K, S)
        tcn_outputs = self.temporal_conv_net(tcn_inputs)
        # (, K, S) -> (, K, C, N)
        estimated_masks = self.output_block(tcn_outputs)
        return estimated_masks

    def get_config(self) -> dict:
        return self.param.get_config()
# ConvTasNetSeparator end


class SeperatorOutputBlock(tf.keras.layers.Layer):
    """Output Process after Dilated-TCN in Separtation Module.

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        prelu (tf.keras.layer.PReLU): PReLU activation layer with a single parameter
        output_conv1x1 (tf.keras.layers.Conv1D): 1x1 convolution layer
        output_reshape (tf.keras.layers.Reshape): (, K, C*N) -> (, K, C, N)    
    """

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(SeperatorOutputBlock, self).__init__(**kwargs)
        self.param = param
        self.prelu = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.output_conv1x1 = tf.keras.layers.Conv1D(filters=self.param.C * self.param.N,
                                                     kernel_size=1,
                                                     activation="sigmoid",
                                                     use_bias=self.param.use_bias)
        self.output_reshape = tf.keras.layers.Reshape(target_shape=(self.param.K,
                                                                    self.param.C,
                                                                    self.param.N))

    def call(self, block_inputs: tf.Tensor) -> tf.Tensor:
        """
        Args:
            block_inputs (tf.Tensor): Tensor of shape=(, K, S)

        Returns:
            block_outputs (tf.Tensor): Tensor of shape=(, K, C, N)
        """
        # (, K, S) -> (, K, S)
        block_outputs = self.prelu(block_inputs)
        # (, K, S) -> (, K, C*N)
        block_outputs = self.output_conv1x1(block_outputs)
        # (, K, C*N) -> (, K, C, N)
        block_outputs = self.output_reshape(block_outputs)
        return block_outputs

    def get_config(self) -> dict:
        return self.param.get_config()
# SeperatorOutputBlock end
