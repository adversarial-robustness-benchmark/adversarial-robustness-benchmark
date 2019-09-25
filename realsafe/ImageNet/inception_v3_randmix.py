from __future__ import absolute_import

import os
from six.moves import urllib
import tarfile

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

from realsafe.model import ClassifierDifferentiable
from realsafe.utils import get_model_path, show_progress

slim = tf.contrib.slim

class Inception_V3_RandMix(ClassifierDifferentiable):
    def __init__(self):
        ClassifierDifferentiable.__init__(self,
                                          x_min=0.0,
                                          x_max=1.0,
                                          x_shape=(299, 299, 3,),
                                          x_dtype=tf.float32,
                                          y_dtype=tf.int32,
                                          n_class=1001)
        self.n_clusters = 5
        self.noise_level = 32.0 / 255.0
        self.num_ensemble = 10

    def logits_and_labels(self, xs_ph):
        xs_ph = xs_ph * 2.0 - 1.0
        batch_size = xs_ph.get_shape().as_list()[0]
        xs_ph_tile = tf.tile(tf.expand_dims(xs_ph, 1), [1, self.num_ensemble, 1, 1, 1])
        xs_ph_tile = tf.reshape(xs_ph_tile, (-1,) + self.x_shape)
        
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            with tf.variable_scope("RandDisc"):
                xs_ph_tile = self.iterative_clustering_layer(source=xs_ph_tile, n_clusters=self.n_clusters,
                                                               sigma=10, alpha=10, noise_level_1=self.noise_level,
                                                               noise_level_2=self.noise_level)
            logits, end_points = inception.inception_v3(
                xs_ph_tile, 
                num_classes=self.n_class, 
                is_training=False, 
                reuse=tf.AUTO_REUSE)
            logits = tf.reshape(logits, [batch_size, self.num_ensemble, -1])
            logits = tf.reduce_mean(logits, axis = 1)
            
            predicted_labels = tf.argmax(logits, 1)
        return logits, predicted_labels

    def labels(self, xs_ph):
        _, labels = self.logits_and_labels(xs_ph)
        return labels

    def load(self, **kwargs):
        session = kwargs["session"]
        assert isinstance(session, tf.Session)

        x_input = tf.placeholder(self.x_dtype, shape=(None,) + self.x_shape)
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            inception.inception_v3(
                x_input,
                num_classes=self.n_class,
                is_training=False,
                reuse=tf.AUTO_REUSE)

        model_path = get_model_path('inception_v3')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            urllib.request.urlretrieve(
                'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz',
                os.path.join(model_path, 'inception_v3_2016_08_28.tar.gz'), show_progress)

            tar = tarfile.open(os.path.join(model_path, 'inception_v3_2016_08_28.tar.gz'))
            file_names = tar.getnames()
            for file_name in file_names:
                tar.extract(file_name, model_path)

        saver = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        saver.restore(session, os.path.join(model_path, 'inception_v3.ckpt'))
    
    def iterative_clustering_layer(self, source, n_clusters, sigma, alpha, noise_level_1, noise_level_2, adaptive_centers=True):
        source1 = source + tf.random_normal(tf.shape(source)) * noise_level_1
        source2 = source + tf.random_normal(tf.shape(source)) * noise_level_2
        if adaptive_centers:
            centroids = self.sample_centroid_with_kpp(source1, 100, n_clusters, sigma)
        else:
            centroids = tf.tile(tf.expand_dims(tf.constant(DIRS, dtype=tf.float32), axis=0), [source.get_shape().as_list()[0], 1, 1]) * 0.5
        image, _ = self.rgb_clustering(source2, centroids, alpha, 0.0)
        return image
    
    def sample_centroid_with_kpp(self, images, n_samples, n_clusters, sigma):
        batchsize, width, height, channel = images.get_shape().as_list()
        images = tf.reshape(images, [-1, width*height, channel])
        samples = []

        for _ in range(n_samples):
            indices = tf.random_uniform(shape=[1], minval=0, maxval=width*height, dtype=tf.int32)
            selected_points = tf.gather(params=images, indices=indices, axis=1)
            samples.append(selected_points)

        samples = tf.concat(samples, axis=1)
        distances = 1e-4 * tf.ones(shape=(batchsize, n_samples))
        centroids = []
        for _ in range(n_clusters):
            indices = tf.reshape(tf.multinomial(sigma * distances, 1), [batchsize])
            weights = tf.expand_dims(tf.one_hot(indices, depth=n_samples), 2)
            selected_points = tf.expand_dims(tf.reduce_sum(weights * samples, axis=1), 1)
            centroids.append(selected_points)
        return tf.concat(centroids, axis=1)
    
    def rgb_clustering(self, images, centroids, alpha, noise_level):
        batchsize, width, height, channel = images.get_shape().as_list()

        # Gaussian mixture clustering
        cluster_num = centroids.get_shape().as_list()[1]
        reshaped_images = tf.reshape(images, [-1, width, height, 1, channel])
        reshaped_centroids = tf.reshape(centroids, [-1, 1, 1, cluster_num, channel])
        distances = tf.reduce_sum(tf.square(reshaped_centroids - reshaped_images), axis=4)
        logits = tf.clip_by_value(-alpha * distances, -200, 200)
        probs = tf.expand_dims(tf.nn.softmax(logits), 4)
        new_images = tf.reduce_sum(reshaped_centroids * probs, axis=3)
        # update cluster centers
        new_centroids = tf.reduce_sum(reshaped_images * probs, axis=[1, 2]) / (tf.reduce_sum(probs, axis=[1, 2]) + 1e-16)
        return new_images, new_centroids
