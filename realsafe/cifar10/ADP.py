from __future__ import absolute_import
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.regularizers import l2
from keras.models import Model

import os
import numpy as np
import tensorflow as tf
from six.moves import urllib

from realsafe.model import ClassifierDifferentiable
from realsafe.utils import get_model_path, show_progress


class ADP(ClassifierDifferentiable):
    def __init__(self):
    	ClassifierDifferentiable.__init__(self,
                                          x_min=0.0,
                                          x_max=1.0,
                                          x_shape=(32, 32, 3,),
                                          x_dtype=tf.float32,
                                          y_dtype=tf.int32,
                                          n_class=10)

    def load(self, **kwargs):
        session = kwargs["session"]
        assert isinstance(session, tf.Session)

        keras.backend.set_session(session)
        model_input = Input(shape=self.x_shape)
        model_dic = {}
        model_out = []
        for i in range(3):
            model_dic[str(i)] = resnet_v2(input=model_input, 
                                          depth=110, 
                                          num_classes=self.n_class)
            model_out.append(model_dic[str(i)][2])
        model_output = keras.layers.concatenate(model_out)
        model = Model(inputs=model_input, outputs=model_output)
        model_ensemble = keras.layers.Average()(model_out)
        model_ensemble = Model(input=model_input, output=model_ensemble)

        model_path = get_model_path('adp-ensemble')
        fname = os.path.join(model_path, 'cifar10_ResNet110v2_model.200.h5')
        if not os.path.exists(fname):
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            urllib.request.urlretrieve(
                "http://ml.cs.tsinghua.edu.cn/~tianyu/ADP/pretrained_models/ADP_standard_3networks/cifar10_ResNet110v2_model.200.h5", 
                fname, show_progress)
        
        datamean = os.path.join(model_path, 'mean.npy')
        
        model.load_weights(fname)
        self.model = model_ensemble
        self.mean = np.load(datamean)

    def logits_and_labels(self, xs_ph):
        prob = self.model(xs_ph - self.mean)
        logits = tf.log(prob)
        predicted_labels = tf.cast(tf.argmax(prob, 1), tf.int32)

        return logits, predicted_labels

    def labels(self, xs_ph):
        prob = self.model(xs_ph - self.mean)
        predicted_labels = tf.cast(tf.argmax(prob, 1), tf.int32)

        return predicted_labels


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input, depth, num_classes=10, dataset='cifar10'):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = input
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    if dataset=='mnist':
        poolsize = 7
    else:
        poolsize = 8
    x = AveragePooling2D(pool_size=poolsize)(x)
    final_features = Flatten()(x)
    logits = Dense(
        num_classes, kernel_initializer='he_normal')(final_features)
    outputs = Activation('softmax')(logits)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model, inputs, outputs, logits, final_features

def resnet_v2(input, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = input
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs, num_filters=num_filters_in, conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_layer(
                inputs=x,
                num_filters=num_filters_in,
                kernel_size=1,
                strides=strides,
                activation=activation,
                batch_normalization=batch_normalization,
                conv_first=False)
            y = resnet_layer(
                inputs=y, num_filters=num_filters_in, conv_first=False)
            y = resnet_layer(
                inputs=y,
                num_filters=num_filters_out,
                kernel_size=1,
                conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    final_features = Flatten()(x)
    logits = Dense(
        num_classes, kernel_initializer='he_normal')(final_features)
    outputs = Activation('softmax')(logits)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model, inputs, outputs, logits, final_features
