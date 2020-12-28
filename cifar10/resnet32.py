# -*- coding: utf-8 -*-

import functools
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers 

class _IdentityBlock(tf.keras.Model):
    """
    该 block 是直通式的 Resnet 模块。
    1.主通道:    x->conv(1,1)->BN-> conv(n,n)->BN-> conv(1,1)->BN
    2.shortcut: x不经过任何处理. 原因是主通道处理后，尺度方面没有变化，可以直接相加

    _IdentityBlock is the block that has no conv layer at shortcut.

    Args:
        kernel_size: the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        data_format: data_format for the input ('channels_first' or
            'channels_last').
    """

    def __init__(self, filters, stage, block):
        super(_IdentityBlock, self).__init__(name='')

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 3

        self.bn2a = layers.BatchNormalization(
                axis=bn_axis, name=bn_name_base + '2a')
        self.conv2a = layers.Conv2D(
                filters, 
                (3, 3), 
                strides=(1, 1), 
                padding='same', 
                name=conv_name_base + '2a',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(0.0001))

        self.bn2b = layers.BatchNormalization(
                axis=bn_axis, name=bn_name_base + '2b')
        self.conv2b = layers.Conv2D(
                filters,
                (3, 3),
                strides=(1, 1),
                padding='same',
                name=conv_name_base + '2b',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(0.0001))

    def call(self, input_tensor, training=False):
        x = self.bn2a(input_tensor, training=training)
        x = tf.nn.relu(x)
        x = self.conv2a(x)

        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2b(x)

        x += input_tensor
        return x


class _ConvBlock(tf.keras.Model):
    """
    该 block 是卷积式的 Resnet 模块。
    1.主通道:    x->conv(1,1)(stride=2)->BN-> conv(n,n)->BN-> conv(1,1)->BN
    2.shortcut: x->conv(1,1)(stride=2). 主通道处理后，尺度缩小为原来的1/2

    _ConvBlock is the block that has a conv layer at shortcut.

    Args:
        kernel_size: the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        data_format: data_format for the input ('channels_first' or
            'channels_last').
        strides: strides for the convolution. Note that from stage 3, the first
         conv layer at main path is with strides=(2,2), and the shortcut should
         have strides=(2,2) as well.
    """

    def __init__(self, filters, stage, block):
        super(_ConvBlock, self).__init__(name='')

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 3

        self.bn2a = layers.BatchNormalization(
                axis=bn_axis, name=bn_name_base + '2a')
        self.conv2a = layers.Conv2D(
                filters, 
                (3, 3),
                padding='same',
                strides=(2, 2),
                name=conv_name_base + '2a',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(0.0001))

        self.bn2b = layers.BatchNormalization(
                axis=bn_axis, name=bn_name_base + '2b')
        self.conv2b = layers.Conv2D(
                filters,
                (3, 3),
                padding='same',
                strides=(1, 1),
                name=conv_name_base + '2b',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(0.0001))

        self.conv_shortcut = layers.Conv2D(
                filters, 
                (1, 1),
                strides= (2, 2),
                padding='same',
                name=conv_name_base + '1',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(0.0001))

    def call(self, input_tensor, training=False):
        x = self.bn2a(input_tensor, training=training)
        x = tf.nn.relu(x)
        x = self.conv2a(x)

        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2b(x)

        shortcut = self.conv_shortcut(input_tensor)

        x += shortcut
        return x


class ResNet32(tf.keras.Model):
    """Instantiates the ResNet50 architecture.

    Args:
        data_format: format for the image. Either 'channels_first' or
            'channels_last'.  'channels_first' is typically faster on GPUs while
            'channels_last' is typically faster on CPUs. See
            https://www.tensorflow.org/performance/performance_guide#data_formats
        name: Prefix applied to names of variables created in the model.
        trainable: Is the model trainable? If true, performs backward
                and optimization after call() method.
        include_top: whether to include the fully-connected layer at the top of the
            network.
        pooling: Optional pooling mode for feature extraction when `include_top`
            is `False`.
            - `None` means that the output of the model will be the 4D tensor
                    output of the last convolutional layer.
            - `avg` means that global average pooling will be applied to the output of
                    the last convolutional layer, and thus the output of the model will be
                    a 2D tensor.
            - `max` means that global max pooling will be applied.
        classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True.

    Raises:
            ValueError: in case of invalid argument for data_format.
    """

    def __init__(self, name='',
                 classes=10):
        super(ResNet32, self).__init__(name=name)

        def conv_block(filters, stage, block):
            # 返回一个残差模块，尺度变小.
            return _ConvBlock(filters,
                    stage=stage, block=block)

        def id_block(filters, stage, block):
            # 返回一个残差模块，尺度不变
            return _IdentityBlock(filters, 
                    stage=stage, block=block)

        # self.conv1 = layers.Conv2D(
        #         16, (3, 3),
        #         strides=(1, 1),
        #         padding='same',
        #         name='conv1',
        #         kernel_initializer='he_normal',
        #         kernel_regularizer=regularizers.l2(0.0001))
        self.conv1 = layers.Conv2D(
                16, (3, 3),
                strides=(1, 1),
                padding='same',
                name='conv1',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(0.0001))
        bn_axis = 3

        self.l2a = id_block(16, stage=2, block='a')
        self.l2b = id_block(16, stage=2, block='b')
        self.l2c = id_block(16, stage=2, block='c')
        self.l2d = id_block(16, stage=2, block='d')
        self.l2e = id_block(16, stage=2, block='e')

        self.l3a = conv_block(32, stage=3, block='a')
        self.l3b = id_block(32, stage=3, block='b')
        self.l3c = id_block(32, stage=3, block='c')
        self.l3d = id_block(32, stage=3, block='d')
        self.l3e = id_block(32, stage=3, block='e')

        self.l4a = conv_block(64, stage=4, block='a')
        self.l4b = id_block(64, stage=4, block='b')
        self.l4c = id_block(64, stage=4, block='c')
        self.l4d = id_block(64, stage=4, block='d')
        self.l4e = id_block(64, stage=4, block='e')

        self.last_bn = layers.BatchNormalization(axis=bn_axis, name='last_bn')
        self.glo_pool = layers.GlobalAveragePooling2D(name='glo_pool')

        self.fc = layers.Dense(classes, name='fc', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))
        self.softmax = layers.Softmax()

    def call(self, inputs, training=True):
        x = self.conv1(inputs)

        x = self.l2a(x, training=training)
        x = self.l2b(x, training=training)
        x = self.l2c(x, training=training)
        x = self.l2d(x, training=training)
        x = self.l2e(x, training=training)

        x = self.l3a(x, training=training)
        x = self.l3b(x, training=training)
        x = self.l3c(x, training=training)
        x = self.l3d(x, training=training)
        x = self.l3e(x, training=training)

        x = self.l4a(x, training=training)
        x = self.l4b(x, training=training)
        x = self.l4c(x, training=training)
        x = self.l4d(x, training=training)
        x = self.l4e(x, training=training)

        x = self.last_bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.glo_pool(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x


class ResNet32_3nn(tf.keras.Model):
    """Instantiates the ResNet50 architecture.

    Args:
        data_format: format for the image. Either 'channels_first' or
            'channels_last'.  'channels_first' is typically faster on GPUs while
            'channels_last' is typically faster on CPUs. See
            https://www.tensorflow.org/performance/performance_guide#data_formats
        name: Prefix applied to names of variables created in the model.
        trainable: Is the model trainable? If true, performs backward
                and optimization after call() method.
        include_top: whether to include the fully-connected layer at the top of the
            network.
        pooling: Optional pooling mode for feature extraction when `include_top`
            is `False`.
            - `None` means that the output of the model will be the 4D tensor
                    output of the last convolutional layer.
            - `avg` means that global average pooling will be applied to the output of
                    the last convolutional layer, and thus the output of the model will be
                    a 2D tensor.
            - `max` means that global max pooling will be applied.
        classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True.

    Raises:
            ValueError: in case of invalid argument for data_format.
    """

    def __init__(self, name='',
                 classes=10):
        super(ResNet32_3nn, self).__init__(name=name)

        def conv_block(filters, stage, block):
            # 返回一个残差模块，尺度变小.
            return _ConvBlock(filters,
                    stage=stage, block=block)

        def id_block(filters, stage, block):
            # 返回一个残差模块，尺度不变
            return _IdentityBlock(filters, 
                    stage=stage, block=block)

        # self.conv1 = layers.Conv2D(
        #         16, (3, 3),
        #         strides=(1, 1),
        #         padding='same',
        #         name='conv1',
        #         kernel_initializer='he_normal',
        #         kernel_regularizer=regularizers.l2(0.0001))
        self.conv1 = layers.Conv2D(
                16, (3, 3),
                strides=(1, 1),
                padding='same',
                name='conv1',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(0.0001))
        bn_axis = 3

        self.l2a = id_block(16, stage=2, block='a')
        self.l2b = id_block(16, stage=2, block='b')
        self.l2c = id_block(16, stage=2, block='c')
        self.l2d = id_block(16, stage=2, block='d')
        self.l2e = id_block(16, stage=2, block='e')

        self.l3a = conv_block(32, stage=3, block='a')
        self.l3b = id_block(32, stage=3, block='b')
        self.l3c = id_block(32, stage=3, block='c')
        self.l3d = id_block(32, stage=3, block='d')
        self.l3e = id_block(32, stage=3, block='e')

        self.l4a = conv_block(64, stage=4, block='a')
        self.l4b = id_block(64, stage=4, block='b')
        self.l4c = id_block(64, stage=4, block='c')
        self.l4d = id_block(64, stage=4, block='d')
        self.l4e = id_block(64, stage=4, block='e')

        self.last_bn = layers.BatchNormalization(axis=bn_axis, name='last_bn')
        self.glo_pool = layers.GlobalAveragePooling2D(name='glo_pool')

        self.fc = layers.Dense(classes, name='fc', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))
        self.softmax = layers.Softmax()

        self.avg_pool = layers.AveragePooling2D((2, 2), name='avg_pool')
        self.last_bn_1 = layers.BatchNormalization(axis=bn_axis, name='last_bn_1')
        self.last_bn_2 = layers.BatchNormalization(axis=bn_axis, name='last_bn_2')
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(classes, name='exit1', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))
        self.dense2 = layers.Dense(classes, name='exit2', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))

    def call(self, inputs, training=True):
        x = self.conv1(inputs)

        x = self.l2a(x, training=training)
        x = self.l2b(x, training=training)
        x = self.l2c(x, training=training)
        x = self.l2d(x, training=training)
        x = self.l2e(x, training=training)

        exit1 = self.last_bn_1(x)
        exit1 = tf.nn.relu(exit1)
        exit1 = self.avg_pool(exit1)
        exit1 = self.flatten(exit1)
        exit1 = self.dense1(exit1)
        exit1 = self.softmax(exit1)

        x = self.l3a(x, training=training)
        x = self.l3b(x, training=training)
        x = self.l3c(x, training=training)
        x = self.l3d(x, training=training)
        x = self.l3e(x, training=training)

        exit2 = self.last_bn_2(x)
        exit2 = tf.nn.relu(exit2)
        exit2 = self.avg_pool(exit2)
        exit2 = self.flatten(exit2)
        exit2 = self.dense2(exit2)
        exit2 = self.softmax(exit2)

        x = self.l4a(x, training=training)
        x = self.l4b(x, training=training)
        x = self.l4c(x, training=training)
        x = self.l4d(x, training=training)
        x = self.l4e(x, training=training)

        x = self.last_bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.glo_pool(x)
        x = self.fc(x)
        x = self.softmax(x)

        return exit1, exit2, x



