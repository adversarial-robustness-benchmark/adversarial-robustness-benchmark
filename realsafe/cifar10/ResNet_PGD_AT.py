from __future__ import absolute_import
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tensorflow as tf
import numpy as np
from six.moves import urllib
from realsafe.model import ClassifierDifferentiable
from realsafe.utils import get_model_path, show_progress

class ResNet_PGD_AT(ClassifierDifferentiable):
    def __init__(self):
        ClassifierDifferentiable.__init__(self,
                                          x_min=0.0,
                                          x_max=255.0,
                                          x_shape=(32, 32, 3,),
                                          x_dtype=tf.float32,
                                          y_dtype=tf.int32,
                                          n_class=10)
        
    def add_internal_summaries(self):
        pass

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def inference(self, xs_ph, reuse=tf.AUTO_REUSE):
        """Build the core model within the graph."""
        with tf.variable_scope('input', reuse=reuse):
            input_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
                                   xs_ph)
            x = self._conv('init_conv', input_standardized, 3, 3, 16, self._stride_arr(1))

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        res_func = self._residual

        filters = [16, 160, 320, 640]

        # Update hps.num_residual_units to 9

        with tf.variable_scope('unit_1_0', reuse=reuse):
            x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                       activate_before_residual[0])
        for i in range(1, 5):
            with tf.variable_scope('unit_1_%d' % i, reuse=reuse):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

        with tf.variable_scope('unit_2_0', reuse=reuse):
            x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                       activate_before_residual[1])
        for i in range(1, 5):
            with tf.variable_scope('unit_2_%d' % i, reuse=reuse):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

        with tf.variable_scope('unit_3_0', reuse=reuse):
            x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                       activate_before_residual[2])
        for i in range(1, 5):
            with tf.variable_scope('unit_3_%d' % i, reuse=reuse):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

        with tf.variable_scope('unit_last', reuse=reuse):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, 0.1)
            x = self._global_avg_pool(x)

        with tf.variable_scope('logit', reuse=reuse):
            self.pre_softmax = self._fully_connected(x, 10)

        self.predictions = tf.argmax(self.pre_softmax, 1)
        return self.pre_softmax
        
    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.name_scope(name):
            return tf.contrib.layers.batch_norm(
              inputs=x,
              decay=.9,
              center=True,
              scale=True,
              activation_fn=None,
              updates_collections=None,
              is_training=False)

    def _residual(self, x, in_filter, out_filter, stride,
                    activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, 0.1)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, 0.1)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, 0.1)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                         [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find('DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        return tf.add_n(costs)

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
              'DW', [filter_size, filter_size, in_filters, out_filters],
              tf.float32, initializer=tf.random_normal_initializer(
                  stddev=np.sqrt(2.0/n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        num_non_batch_dimensions = len(x.shape)
        prod_non_batch_dimensions = 1
        for ii in range(num_non_batch_dimensions - 1):
            prod_non_batch_dimensions *= int(x.shape[ii + 1])
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        w = tf.get_variable(
            'DW', [prod_non_batch_dimensions, out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2]) 
        
    def logits_and_labels(self, xs_ph):
        logits =self.inference(
            xs_ph, 
            reuse=tf.AUTO_REUSE)
        predicts = tf.nn.softmax(logits)
        predicted_labels = tf.cast(tf.argmax(predicts, 1), tf.int32)
        return logits, predicted_labels

    def labels(self, xs_ph):
        _, labels = self.logits_and_labels(xs_ph)
        return labels

    def load(self, **kwargs):
        session = kwargs["session"]
        assert isinstance(session, tf.Session)

        x_input = tf.placeholder(tf.float32, shape=(None,) + self.x_shape)
        logits = self.inference(
            x_input, 
            reuse=False)
        model_path = get_model_path('pgd_at')
        url = 'https://www.dropbox.com/s/g4b6ntrp8zrudbz/adv_trained.zip?dl=1'
        fname = os.path.join(model_path, url.split('/')[-1])
        if not os.path.exists(fname):
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            
            from six.moves import urllib
            urllib.request.urlretrieve(url, fname)
            import zipfile
            model_zip = zipfile.ZipFile(fname)
                
            model_zip.extractall(model_path)
            print('Extracted model in {}'.format(model_zip.namelist()[1]))
       
        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint(model_path+'/adv_trained')
        saver.restore(session, checkpoint)
        
    
