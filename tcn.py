"""
Temporal Convolution Network

Authors:
    kaparoo
"""

import tensorflow as tf
from conv_tasnet import ConvTasNetParam


class Conv1DBlock(tf.keras.layers.Layer):
    def __init__(self, param: ConvTasNetParam, dilation: int = 1, **kwargs):
        super(Conv1DBlock, self).__init__(**kwargs)
        self.param = param
        self.dilation = dilation
        self.bottleneck_conv1x1 = tf.keras.layers.Conv1D(filters=self.param.H,
                                                         kernel_size=1,
                                                         use_bias=False)
        self.prelu1 = tf.keras.layers.PReLU()
        self.depthwise_conv = tf.keras.layers.Conv1D(filters=self.param.H,
                                                     kernel_size=1,
                                                     dilation_rate=self.dilation,
                                                     groups=self.param.H,
                                                     padding='same',
                                                     use_bias=False)
        self.prelu2 = tf.keras.layers.PReLU()
        self.skip_conv1x1 = tf.keras.layers.Conv1D(filters=self.param.Sc,
                                                   kernel_size=1,
                                                   use_bias=False)
        self.residaul_conv1x1 = tf.keras.layers.Conv1D(filters=self.param.B,
                                                       kernel_size=1,
                                                       use_bias=False)

    def call(self, block_inputs):
        """
        Args:
            block_inputs: [T_hat x B]

        Returns:
            residual_outputs: [T_hat x B]
            skip_outpus: [T_hat x Sc]
        """
        # [T_hat x B] -> [T_hat x H]
        block_outputs = self.bottleneck_conv1x1(block_inputs)
        # [T_hat x H] -> [T_hat x H]
        block_outputs = self.prelu1(block_outputs)
        # [T_hat x H] -> [T_hat x H]
        block_outputs = self.depthwise_conv(block_outputs)
        # [T_hat x H] -> [T_hat x H]
        block_outputs = self.prelu2(block_outputs)
        # [T_hat x H] -> [T_hat x Sc]
        skip_outputs = self.skip_conv1x1(block_outputs)
        # [T_hat x H] -> [T_hat x B]
        residual_outputs = block_inputs + self.residaul_conv1x1(block_outputs)
        return residual_outputs, skip_outputs

    def get_config(self):
        return {**self.param.get_config(),
                'dilation': self.dilation}
# Conv1DBlock end


class TemporalConvNet(tf.keras.layers.Layer):
    """
    Dilated Temporal Convolutional Network

    Attributes:
        param (ConvTasNetParam): Hyperparameters
    """

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(TemporalConvNet, self).__init__(**kwargs)
        self.param = param
        self.conv1d_stack = []
        for _ in range(self.param.R):
            for x in range(self.param.X):
                self.conv1d_stack.append(
                    Conv1DBlock(self.param, dilation=2**x))

    def call(self, tcn_inputs):
        """
        Args:
            tcn_inputs: [T_hat x B]

        Returns:
            tcn_outputs: [T_hat x Sc]
        """
        tcn_outputs = None
        for conv1d_block in self.conv1d_stack:
            tcn_inputs, skip_outputs = conv1d_block(tcn_inputs)
            if tcn_outputs != None:
                tcn_outputs = tcn_outputs + skip_outputs
            else:
                tcn_outputs = skip_outputs

        return tcn_outputs

    def get_config(self):
        return self.param.get_config()
# TemporalConvNet end
