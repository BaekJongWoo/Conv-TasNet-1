import tensorflow as tf

from .param import ConvTasNetParam


class Encoder(tf.keras.layers.Layer):

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(Encoder, self).__init__(name='Encoder', **kwargs)

        self.U = tf.keras.layers.Conv1D(filters=param.N,
                                        kernel_size=1,
                                        activation='linear',
                                        use_bias=False)

    def call(self, mixture_segments):
        # (, That, L) -> (, That, N)
        return self.U(mixture_segments)


class Decoder(tf.keras.layers.Layer):

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(Decoder, self).__init__(name='Decoder', **kwargs)

        self.B = tf.keras.layers.Conv1D(filters=param.L,
                                        kernel_size=1,
                                        activation='sigmoid',
                                        use_bias=False)

    def call(self, source_weights):
        # (, C, That, N) -> (, C, That, L)
        return self.B(source_weights)


class Separater(tf.keras.layers.Layer):

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(Separater, self).__init__(name='Separation', **kwargs)

        self.layer_normalization = tf.keras.layers.LayerNormalization()

        self.conv1x1_1 = tf.keras.layers.Conv1D(filters=param.B,
                                                kernel_size=1,
                                                use_bias=False)

        # Dilated-TCN
        self.conv1d_blocks = []
        for r in range(param.R):
            for x in range(param.X):
                self.conv1d_blocks.append(Conv1DBlock(param, r, x))
        self.conv1d_blocks[-1].is_last = True
        self.skip_connection = tf.keras.layers.Add()

        self.prelu = tf.keras.layers.PReLU(shared_axes=[1, 2])

        self.conv1x1_2 = tf.keras.layers.Conv1D(filters=param.C*param.N,
                                                kernel_size=1,
                                                acivation='sigmoid',
                                                use_bias=False)

        self.reshape_mask = tf.keras.layers.Reshape(
            target_shape=[param.That, param.C, param.N])

        self.reorder_mask = tf.keras.layers.Permute([2, 1, 3])

    def call(self, mixture_weights):
        # (, That, N) -> (, That, N)
        normalized_weights = self.layer_normalization(mixture_weights)

        # (, That, N) -> (, That, B)
        block_inputs = self.conv1x1_1(normalized_weights)

        # (, That, B) -> (, That, Sc)
        skip_outputs = []
        for conv1d_block in self.conv1d_blocks:
            _skip_outputs, _block_outputs = conv1d_block(block_inputs)
            block_inputs = _block_outputs
            skip_outputs.append(_skip_outputs)
        tcn_outputs = self.skip_connection(skip_outputs)
        tcn_outputs = self.prelu(tcn_outputs)

        # (, That, Sc) -> (, That, C*N)
        source_masks = self.conv1x1_2(tcn_outputs)

        # (, That, C*N) -> (, That, C, N)
        source_masks = self.reshape_mask(source_masks)

        # (, That, C, N) -> (, C, That, N)
        return self.reorder_mask(source_masks)


class Conv1DBlock(tf.keras.layers.Layer):

    def __init__(self, param: ConvTasNetParam, r: int, x: int, **kwargs):
        super(Conv1DBlock, self).__init__(
            name=f'conv1d_block_r{r}_x{x}', **kwargs)

        self.is_last = False
        self.conv1x1_bottle = tf.keras.layers.Conv1D(param.H)
        self.prelu1 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.conv1x1_skipconn = tf.keras.layers.Conv1D()
        self.conv1x1_residual = tf.keras.layers.Conv1D()
        self.prelu2 = tf.keras.layers.PReLU(shared_axes=[1, 2])
