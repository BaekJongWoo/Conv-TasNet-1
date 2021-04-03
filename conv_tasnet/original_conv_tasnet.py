import tensorflow as tf
from .conv_tasnet_param import ConvTasNetParam
from .temporal_conv_net import ConvTasNetTCN


class ConvTasNet(tf.keras.Model):
    """Conv-TasNet Model

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
        self.decoder = ConvTasNetDecoder(self.param)
        self.separator = ConvTasNetSeparator(self.param)

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
    """Encoding Module using 1-D Convolution

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        conv1d_U (tf.keras.layers.Conv1D): 1-D convolution layer
    """

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetEncoder, self).__init__(**kwargs)
        self.param = param
        self.conv1d_U = tf.keras.layers.Conv1D(filters=self.param.N,
                                               kernel_size=1,
                                               activation="linear",
                                               use_bias=False)

    def call(self, mixture_segments: tf.Tensor) -> tf.Tensor:
        """
        Args:
            mixture_segments (tf.Tensor): Tensor of shape=(, K, L)

        Returns:
            mixture_weights (tf.Tensor): Tensor of shape=(, K, N) 
        """
        # (, K, L) -> (, K, N)
        mixture_weights = self.conv1d_U(mixture_segments)
        return mixture_weights

    def get_config(self) -> dict:
        return self.param.get_config()
# ConvTasNetEncoder end


class ConvTasNetDecoder(tf.keras.layers.Layer):
    """Decoding Module using 1-D Convolution

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
                                              use_bias=False)
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
        sources_weights = tf.expand_dims(estimated_masks, 2)
        # (, K, 1, N), (, K, C, N) -> (, K, C, N)
        sources_weights = self.multiply([mixture_weights, sources_weights])
        # (, K, C, N) -> (, K, C, L)
        estimated_sources = self.conv1_V(sources_weights)
        # (, K, C, L) -> (, C, K, L)
        estimated_sources = self.permute(estimated_sources)
        return estimated_sources

    def get_config(self) -> dict:
        return self.param.get_config()
# ConvTasNetDecoder end


class ConvTasNetSeparator(tf.keras.layers.Layer):
    """Separation Module using Dilated Temporal Convolution Network (Dilated-TCN)

    Attributes:
        param (ConvTasNetParam): Hyperparameters
        normalization (tf.keras.layers.LayerNormalization): Normalization layer
        input_conv1x1 (tf.keras.layers.Conv1D): 1x1 convolution layer
        temporal_conv_net (ConTasNetTCN): Dilated-TCN layer
        prelu (tf.keras.layer.PReLU): PReLU activation layer
        output_conv1x1 (tf.keras.layers.Conv1D): 1x1 convolution layer
        output_reshape (tf.keras.layers.Reshape): (, K, C*N) -> (, K, C, N)
        sigmoid (tf.keras.activations.sigmoid): Sigmoid activation
    """

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(ConvTasNetSeparator, self).__init__(**kwargs)
        self.param = param
        self.normalization = tf.keras.layers.LayerNormalization(self.param.eps)
        self.input_conv1x1 = tf.keras.layers.Conv1D(filters=self.param.B,
                                                    kernel_size=1,
                                                    use_bias=False)
        self.temporal_conv_net = ConvTasNetTCN(self.param)
        self.prelu = tf.keras.layers.PReLU()
        self.output_conv1x1 = tf.keras.layers.Conv1D(filters=self.param.C * self.param.N,
                                                     kernel_size=1,
                                                     use_bias=False)
        self.output_reshape = tf.keras.layers.Reshape(target_shape=(self.param.K,
                                                                    self.param.C,
                                                                    self.param.N))
        self.sigmoid = tf.keras.activations.sigmoid

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
        # (, K, S) -> (, K, S)
        tcn_outputs = self.prelu(tcn_outputs)
        # (, K, S) -> (, K, C*N)
        tcn_outputs = self.output_conv1x1(tcn_outputs)
        # (, K, C*N) -> (, K, C, N)
        estimated_masks = self.output_reshape(tcn_outputs)
        # (, K, C, N) -> (, K, C, N)
        estimated_masks = self.sigmoid(estimated_masks)
        return estimated_masks

    def get_config(self):
        return self.param.get_config()
# ConvTasNetSeparator end