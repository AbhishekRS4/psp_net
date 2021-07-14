# @author : Abhishek R S

import os
import h5py
import numpy as np
import tensorflow as tf

"""
PSPNet
# Reference
- [Deep Residual Learning for Image Recognition]
  (https://arxiv.org/abs/1512.03385)
- [PSPNet](https://arxiv.org/pdf/1612.01105.pdf)
- [PSPNet Project](https://hszhao.github.io/projects/pspnet/index.html)

# Pretrained model weights
- [Download pretrained resnet-50 model]
  (https://github.com/fchollet/deep-learning-models/releases/)
"""

class PSPNet:
    def __init__(self, pretrained_weights, is_training, data_format="channels_first", num_classes=15):
        self._weights_h5 = h5py.File(pretrained_weights, "r")
        self._is_training = is_training
        self._data_format = data_format
        self._num_classes = num_classes
        self._padding = "SAME"
        self._feature_map_axis = None
        self._encoder_data_format = None
        self._encoder_pool_kernel = None
        self._input_size = [512, 1024]
        self._avg_pool_kernel_strides_map = {1: 60, 2: 30, 3: 20, 6: 10}
        self._encoder_conv_strides = [1, 1, 1, 1]
        self._encoder_pool_strides = None
        self._initializer = tf.contrib.layers.xavier_initializer_conv2d()

        """
        based on the data format set appropriate pool_kernel and pool_strides
        always use channels_first i.e. NCHW as the data format on a GPU
        """

        if data_format == "channels_first":
            self._encoder_data_format = "NCHW"
            self._encoder_pool_kernel = [1, 1, 3, 3]
            self._encoder_pool_strides = [1, 1, 2, 2]
            self._feature_map_axis = 1
        else:
            self._encoder_data_format = "NHWC"
            self._encoder_pool_kernel = [1, 3, 3, 1]
            self._encoder_pool_strides = [1, 2, 2, 1]
            self._feature_map_axis = -1

    # build resnet-50 encoder
    def resnet50_encoder(self, features):

        # input : BGR format with image_net mean subtracted
        # bgr mean : [103.939, 116.779, 123.68]

        if self._data_format == "channels_last":
            features = tf.transpose(features, perm=[0, 2, 3, 1])

        # Stage 0
        self.stage0 = self._res_conv_layer(features, "conv1", strides=self._encoder_pool_strides)
        self.stage0 = self._get_batchnorm_layer(self.stage0, "bn_conv1")
        self.stage0 = tf.nn.relu(self.stage0, name="relu1")

        # Stage 1
        self.stage1 = tf.nn.max_pool(
            self.stage0, ksize=self._encoder_pool_kernel, strides=self._encoder_pool_strides,
            padding=self._padding, data_format=self._encoder_data_format, name="pool1"
        )

        # Stage 2
        self.stage2 = self._res_conv_block(input_layer=self.stage1, stage="2a", strides=self._encoder_conv_strides)
        self.stage2 = self._res_identity_block(input_layer=self.stage2, stage="2b")
        self.stage2 = self._res_identity_block(input_layer=self.stage2, stage="2c")

        # Stage 3
        self.stage3 = self._res_conv_block(input_layer=self.stage2, stage="3a", strides=self._encoder_pool_strides)
        self.stage3 = self._res_identity_block(input_layer=self.stage3, stage="3b")
        self.stage3 = self._res_identity_block(input_layer=self.stage3, stage="3c")
        self.stage3 = self._res_identity_block(input_layer=self.stage3, stage="3d")

        # Stage 4
        self.stage4 = self._res_conv_block(input_layer=self.stage3, stage="4a", strides=self._encoder_conv_strides)
        self.stage4 = self._res_identity_block(input_layer=self.stage4, stage="4b")
        self.stage4 = self._res_identity_block(input_layer=self.stage4, stage="4c")
        self.stage4 = self._res_identity_block(input_layer=self.stage4, stage="4d")
        self.stage4 = self._res_identity_block(input_layer=self.stage4, stage="4e")
        self.stage4 = self._res_identity_block(input_layer=self.stage4, stage="4f")

        # Stage 5
        self.stage5 = self._res_conv_block(input_layer=self.stage4, stage="5a", strides=self._encoder_conv_strides)
        self.stage5 = self._res_identity_block(input_layer=self.stage5, stage="5b")
        self.stage5 = self._res_identity_block(input_layer=self.stage5, stage="5c")

    # build psp-net decoder
    def psp_net_decoder(self):
        self.spp_out = self._spatial_pyramid_pool_block(self.stage5, name="spp_")

        self.decoder_conv = self._get_conv2d_layer(
            self.spp_out, 512, [3, 3], [1, 1], use_bias=False, name="decoder_conv"
        )
        self.decoder_bn = self._get_batchnorm_layer(self.decoder_conv, name="decoder_bn")
        self.decoder_elu = self._get_elu_activation(self.decoder_bn, name="decoder_elu")
        self.decoder_dropout = self._get_dropout_layer(self.decoder_elu, rate=0.1, name="decoder_dropout")

        self.decoder_final_conv = self._get_conv2d_layer(
            self.decoder_dropout, self._num_classes, [1, 1], [1, 1], name="decoder_final_conv"
        )

        self.logits = self._get_upsample_layer(
            self.decoder_final_conv, target_size=self._input_size, name="upsample8"
        )

    # build pyramid pooling block
    def _spatial_pyramid_pool_block(self, input_layer, name="spp_"):
        _interp_block1 = self.interp_block(input_layer, 1, name=name + "interp_block1_")
        _interp_block2 = self.interp_block(input_layer, 2, name=name + "interp_block2_")
        _interp_block3 = self.interp_block(input_layer, 3, name=name + "interp_block3_")
        _interp_block6 = self.interp_block(input_layer, 6, name=name + "interp_block6_")

        _concat_features = tf.concat(
            [input_layer, _interp_block1, _interp_block2, _interp_block3, _interp_block6],
            axis=self._feature_map_axis, name=name + "concat"
        )

        return _concat_features

    # build interpolation block
    def interp_block(self, input_layer, level, name):
        if self._data_format == "channels_first":
            _feature_map_size = tf.shape(input_layer)[2:]
        else:
            _feature_map_size = tf.shape(input_layer)[1:3]

        _avg_pool = self._get_avg_pool_layer(
            input_layer, self._avg_pool_kernel_strides_map[level], self._avg_pool_kernel_strides_map[level], name=name + "avg_pool"
        )
        _conv = self._get_conv2d_layer(_avg_pool, 512, [1, 1], [1, 1], use_bias=False, name=name + "conv")
        _bn = self._get_batchnorm_layer(_conv, name=name + "bn")
        _elu = self._get_elu_activation(_bn, name=name + "elu")

        _interp = self._get_upsample_layer(_elu, target_size=_feature_map_size, name=name + "up")

        return _interp

    # return convolution2d layer
    def _get_conv2d_layer(self, input_layer, num_filters, kernel_size, strides, use_bias=True, name="conv"):
        conv_2d_layer = tf.layers.conv2d(
            inputs=input_layer, filters=num_filters, kernel_size=kernel_size, strides=strides, use_bias=use_bias,
            padding=self._padding, data_format=self._data_format, kernel_initializer=self._initializer, name=name
        )
        return conv_2d_layer

    # return avg pool layer
    def _get_avg_pool_layer(self, input_layer, pool_size, strides, name="avg_pool"):
        avg_pool_layer = tf.layers.average_pooling2d(input_layer, pool_size, strides, data_format=self._data_format, name=name)
        return avg_pool_layer

    # return upsample layer
    def _get_upsample_layer(self, input_layer, target_size, name="upsample"):
        if self._data_format == "channels_first":
            input_layer = tf.transpose(input_layer, perm=[0, 2, 3, 1])

        upsampled = tf.image.resize_bilinear(input_layer, target_size, name=name)

        if self._data_format == "channels_first":
            upsampled = tf.transpose(upsampled, perm=[0, 3, 1, 2])

        return upsampled

    # return elu activation function
    def _get_elu_activation(self, input_layer, name="elu"):
        return tf.nn.elu(input_layer, name=name)

    # return dropout layer
    def _get_dropout_layer(self, input_layer, rate=0.5, name="dropout"):
        dropout_layer = tf.layers.dropout(inputs=input_layer, rate=rate, training=self._is_training, name=name)
        return dropout_layer

    # return batch normalization layer
    def _get_batchnorm_layer(self, input_layer, name="bn"):
        bn_layer = tf.layers.batch_normalization(
            input_layer, axis=self._feature_map_axis, training=self._is_training, name=name
        )
        return bn_layer

    #---------------------------------------#
    # pretrained resnet50 encoder functions #
    #---------------------------------------#
    #-----------------------#
    # convolution layer     #
    #-----------------------#
    def _res_conv_layer(self, input_layer, name, strides=[1, 1, 1, 1]):
        W_init_value = np.array(self._weights_h5[name][name + "_W_1:0"], dtype=np.float32)
        b_init_value = np.array(self._weights_h5[name][name + "_b_1:0"], dtype=np.float32)

        W = tf.get_variable(name=name + "kernel", shape=W_init_value.shape,
            initializer=tf.constant_initializer(W_init_value), dtype=tf.float32
        )
        b = tf.get_variable(name=name + "bias", shape=b_init_value.shape,
            initializer=tf.constant_initializer(b_init_value), dtype=tf.float32
        )

        x = tf.nn.conv2d(input_layer, filter=W, strides=strides, padding=self._padding,
            data_format=self._encoder_data_format, name=name + "_conv"
        )
        x = tf.nn.bias_add(x, b, data_format=self._encoder_data_format, name=name + "_bias")

        return x

    #-----------------------#
    # convolution block     #
    #-----------------------#
    def _res_conv_block(self, input_layer, stage, strides):
        x = self._res_conv_layer(input_layer, name="res" + stage + "_branch2a", strides=strides)
        x = self._get_batchnorm_layer(x, name="bn" + stage + "_branch2a")
        x = tf.nn.relu(x, name="relu" + stage + "_branch2a")

        x = self._res_conv_layer(x, name="res" + stage + "_branch2b")
        x = self._get_batchnorm_layer(x, name="bn" + stage + "_branch2b")
        x = tf.nn.relu(x, name="relu" + stage + "_branch2b")

        x = self._res_conv_layer(x, name="res" + stage + "_branch2c")
        x = self._get_batchnorm_layer(x, name="bn" + stage + "_branch2c")

        shortcut = self._res_conv_layer(input_layer, name="res" + stage + "_branch1", strides=strides)
        shortcut = self._get_batchnorm_layer(shortcut, name="bn" + stage + "_branch1")

        x = tf.add(x, shortcut, name="add" + stage)
        x = tf.nn.relu(x, name="relu" + stage)

        return x

    #-----------------------#
    # identity block        #
    #-----------------------#
    def _res_identity_block(self, input_layer, stage):
        x = self._res_conv_layer(input_layer, name="res" + stage + "_branch2a")
        x = self._get_batchnorm_layer(x, name="bn" + stage + "_branch2a")
        x = tf.nn.relu(x, name="relu" + stage + "_branch2a")

        x = self._res_conv_layer(x, name="res" + stage + "_branch2b")
        x = self._get_batchnorm_layer(x, name="bn" + stage + "_branch2b")
        x = tf.nn.relu(x, name="relu" + stage + "_branch2b")

        x = self._res_conv_layer(x, name="res" + stage + "_branch2c")
        x = self._get_batchnorm_layer(x, name="bn" + stage + "_branch2c")

        x = tf.add(x, input_layer, name="add" + stage)
        x = tf.nn.relu(x, name="relu" + stage)

        return x
