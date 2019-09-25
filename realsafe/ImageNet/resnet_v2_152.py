from __future__ import absolute_import

import os
from six.moves import urllib
import tarfile

import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2

from realsafe.model import ClassifierDifferentiable
from realsafe.utils import get_model_path, show_progress

slim = tf.contrib.slim

class ResNet_V2_152(ClassifierDifferentiable):
    def __init__(self):
        ClassifierDifferentiable.__init__(self,
                                          x_min=0.0,
                                          x_max=1.0,
                                          x_shape=(299, 299, 3,),
                                          x_dtype=tf.float32,
                                          y_dtype=tf.int32,
                                          n_class=1001)


    def logits_and_labels(self, xs_ph):
        xs_ph = xs_ph * 2.0 - 1.0
        
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_152(
                xs_ph, 
                num_classes=self.n_class, 
                is_training=False, 
                reuse=tf.AUTO_REUSE)

            predicted_labels = tf.argmax(end_points['predictions'], 1)

        return logits, predicted_labels

    def labels(self, xs_ph):
        _, labels = self.logits_and_labels(xs_ph)
        return labels


    def load(self, **kwargs):
        session = kwargs["session"]
        assert isinstance(session, tf.Session)

        x_input = tf.placeholder(self.x_dtype, shape=(None,) + self.x_shape)
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            resnet_v2.resnet_v2_152(
                x_input,
                num_classes=self.n_class,
                is_training=False,
                reuse=tf.AUTO_REUSE)

        model_path = get_model_path('resnet_v2_152')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            urllib.request.urlretrieve(
                'http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz',
                os.path.join(model_path, 'resnet_v2_152_2017_04_14.tar.gz'), show_progress)

            tar = tarfile.open(os.path.join(model_path, 'resnet_v2_152_2017_04_14.tar.gz'))
            file_names = tar.getnames()
            for file_name in file_names:
                tar.extract(file_name, model_path)

        saver = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))
        saver.restore(session, os.path.join(model_path, 'resnet_v2_152.ckpt'))