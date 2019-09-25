from __future__ import absolute_import
import os

import tensorflow as tf
import numpy as np
from six.moves import urllib
from tensorflow.contrib.slim.nets import resnet_v2

from realsafe.model import ClassifierDifferentiable
from realsafe.utils import get_model_path, show_progress
slim = tf.contrib.slim

class Resnetv2_ALP(ClassifierDifferentiable):
    def __init__(self):
        ClassifierDifferentiable.__init__(self,
                                          x_min=0.0,
                                          x_max=1.0,
                                          x_shape=(64, 64, 3,),
                                          x_dtype=tf.float32,
                                          y_dtype=tf.int32,
                                          n_class=1001)
        
    
        
    def logits_and_labels(self, xs_ph):
    
        xs_ph = xs_ph * 2.0 - 1.0
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, _ = resnet_v2.resnet_v2_50(xs_ph, self.n_class, is_training=False,
                              reuse=tf.AUTO_REUSE)
        logits = tf.squeeze(logits, [1, 2])
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
        with tf.contrib.framework.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, _ = resnet_v2.resnet_v2_50(x_input, self.n_class, is_training=False,
                              reuse=tf.AUTO_REUSE)
        
        model_path = get_model_path('alp')
        url = 'http://download.tensorflow.org/models/adversarial_logit_pairing/imagenet64_alp025_2018_06_26.ckpt.tar.gz'
        fname = os.path.join(model_path, url.split('/')[-1])
        if not os.path.exists(fname):
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            
            from six.moves import urllib
            urllib.request.urlretrieve(url, fname, show_progress)
            import tarfile 
            t = tarfile.open(fname)
            t.extractall(model_path)
            print('Extracted model')
        
        saver = tf.train.Saver()
        saver.restore(session, fname.split('.tar.gz')[0])
        
    
