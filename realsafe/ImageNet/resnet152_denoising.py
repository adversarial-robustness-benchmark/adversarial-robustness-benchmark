from __future__ import absolute_import
import os

import tensorflow as tf
import numpy as np
from six.moves import urllib
from realsafe.model import ClassifierDifferentiable
from realsafe.utils import get_model_path, show_progress

from tensorpack import TowerContext
from tensorpack.tfutils import get_model_loader
from tensorpack.tfutils.argscope import argscope
from tensorpack.models import (
    Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm, FullyConnected, BNReLU)

class ResNet152_Denoising(ClassifierDifferentiable):
    def __init__(self):
        ClassifierDifferentiable.__init__(self,
                                          x_min=0.0,
                                          x_max=1.0,
                                          x_shape=(224, 224, 3,),
                                          x_dtype=tf.float32,
                                          y_dtype=tf.int32,
                                          n_class=1000)
        
        self.model = ResNetDenoiseModel()
        
    def logits_and_labels(self, xs_ph):
        xs_ph = xs_ph * 2.0 - 1.0
        xs_ph = xs_ph[:,:,:,::-1]
        xs_ph = tf.transpose(xs_ph, [0, 3, 1, 2])
        with TowerContext('', is_training=False):
            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                logits = self.model.get_logits(xs_ph)
        predicts = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(predicts, 1)
        return logits, predicted_labels

    def labels(self, xs_ph):
        _, labels = self.logits_and_labels(xs_ph)
        return labels

    def load(self, **kwargs):
        session = kwargs["session"]
        assert isinstance(session, tf.Session)
        
        x_input = tf.placeholder(tf.float32, shape=(None,) + self.x_shape)
        x_input = tf.transpose(x_input, [0, 3, 1, 2]) 
        with TowerContext('', is_training=False):
            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                logits = self.model.get_logits(x_input) 
        
        model_path = get_model_path('R152-Denoise.npz')
        url = 'https://github.com/facebookresearch/ImageNet-Adversarial-Training/releases/download/v0.1/R152-Denoise.npz'
        if not os.path.exists(model_path):
            if not os.path.exists(os.path.dirname(model_path)):
                os.makedirs(os.path.dirname(model_path))
            
            from six.moves import urllib
            urllib.request.urlretrieve(url, model_path, show_progress)
           
        get_model_loader(model_path).init(session)
  
class ResNetDenoiseModel(object):
    def __init__(self):
        self.num_blocks = [3, 8, 36, 3]

    def get_logits(self, image):
        def group_func(name, *args):
            l = resnet_group(name, *args)
            l = denoising(name + '_denoise', l, embed=True, softmax=True)
            return l
        return resnet_backbone(image, self.num_blocks, group_func, resnet_bottleneck)

def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    n_in = l.get_shape().as_list()[1]
    if n_in != n_out:  
        return Conv2D('convshortcut', l, n_out, 1, strides=stride, activation=activation)
    else:
        return l

def get_bn(zero_init=False):
    if zero_init:
        return lambda x, name=None: BatchNorm('bn', x, gamma_initializer=tf.zeros_initializer())
    else:
        return lambda x, name=None: BatchNorm('bn', x)

def resnet_bottleneck(l, ch_out, stride, group=1, res2_bottleneck=64):
    ch_factor = res2_bottleneck * group // 64
    shortcut = l
    l = Conv2D('conv1', l, ch_out * ch_factor, 1, strides=1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out * ch_factor, 3, strides=stride, activation=BNReLU, split=group)
    
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
    
    ret = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(ret, name='block_output')

def resnet_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                current_stride = stride if i == 0 else 1
                l = block_func(l, features, current_stride)
    return l

def resnet_backbone(image, num_blocks, group_func, block_func):
    with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm], data_format='NCHW'), \
            argscope(Conv2D, use_bias=False,
                     kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        l = Conv2D('conv0', image, 64, 7, strides=2, activation=BNReLU)
        l = MaxPooling('pool0', l, pool_size=3, strides=2, padding='SAME')
        l = group_func('group0', l, block_func, 64, num_blocks[0], 1)
        l = group_func('group1', l, block_func, 128, num_blocks[1], 2)
        l = group_func('group2', l, block_func, 256, num_blocks[2], 2)
        l = group_func('group3', l, block_func, 512, num_blocks[3], 2)
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, 1000,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        
    return logits

def denoising(name, l, embed=True, softmax=True):
    with tf.variable_scope(name):
        f = non_local_op(l, embed=embed, softmax=softmax)
        f = Conv2D('conv', f, l.shape[1], 1, strides=1, activation=get_bn(zero_init=True))
        l = l + f
    return l

def non_local_op(l, embed, softmax):
    n_in, H, W = l.shape.as_list()[1:]
    if embed:
        theta = Conv2D('embedding_theta', l, n_in / 2, 1,
                       strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        phi = Conv2D('embedding_phi', l, n_in / 2, 1,
                     strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        g = l
    else:
        theta, phi, g = l, l, l
    if n_in > H * W or softmax:
        f = tf.einsum('niab,nicd->nabcd', theta, phi)
        if softmax:
            orig_shape = tf.shape(f)
            f = tf.reshape(f, [-1, H * W, H * W])
            f = f / tf.sqrt(tf.cast(theta.shape[1], theta.dtype))
            f = tf.nn.softmax(f)
            f = tf.reshape(f, orig_shape)
        f = tf.einsum('nabcd,nicd->niab', f, g)
    else:
        f = tf.einsum('nihw,njhw->nij', phi, g)
        f = tf.einsum('nij,nihw->njhw', f, theta)
    if not softmax:
        f = f / tf.cast(H * W, f.dtype)
    return tf.reshape(f, tf.shape(l))

    
