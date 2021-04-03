# -*- coding: utf-8 -*-

"""Tensorflow 2.x (with Keras API) Implementation of the Fully-Convolutional Time-domain Audio Separation Network (Conv-TasNet)

Authors:
    kaparoo (kaparoo2001@gmail.com)

References:
    [1] Y. Luo and N. Mesgarani, "TaSNet: Time-Domain Audio Separation Network for Real-Time, Single-Channel Speech Separation,"
        2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Calgary, AB, Canada, 2018, pp. 696-700,
        doi: 10.1109/ICASSP.2018.8462116.
    [2] Y. Luo and N. Mesgarani, "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation,"
        in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 27, no. 8, pp. 1256-1266, Aug. 2019,
        doi: 10.1109/TASLP.2019.2915167.
    [3] PyTorch implementation 1: https://github.com/naplab/Conv-TasNet
    [4] PyTorch implementation 2: https://github.com/kaituoxu/Conv-TasNet
    [5] TasNet implementaion: https://github.com/paxbun/TasNet 

Notices:
    For every tf.Tensor variable, notation (, K, N, ...) means that the variable is the tf.Tensor of shape=(Batch_size, K, N, ...)
"""

__all__ = ["ConvTasNetParam", "ConvTasNet", "SISNR", "SDR"]

from .conv_tasnet_param import ConvTasNetParam
from .original_conv_tasnet import ConvTasNet
from .losses import SISNR, SDR
