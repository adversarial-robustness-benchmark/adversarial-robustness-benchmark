from __future__ import absolute_import
import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Classifier(object):
    """
    x_shape is the shape tuple of classifier input, for example (784,) for MNIST classifiers.
    """

    def __init__(self, x_min, x_max, x_shape, x_dtype, y_dtype, n_class):
        self.x_min = x_min
        self.x_max = x_max
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.n_class = n_class

    @abc.abstractmethod
    def labels(self, xs_ph):
        """
        :param xs_ph: tf.Tensor
        :return: tf.Tensor
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, **kwargs):
        raise NotImplementedError


@six.add_metaclass(abc.ABCMeta)
class ClassifierWithLogits(Classifier):
    def __init__(self, x_min, x_max, x_shape, x_dtype, y_dtype, n_class):
        Classifier.__init__(self,
                            x_min=x_min,
                            x_max=x_max,
                            x_shape=x_shape,
                            x_dtype=x_dtype,
                            y_dtype=y_dtype,
                            n_class=n_class)

    @abc.abstractmethod
    def logits_and_labels(self, xs_ph):
        """
        :param xs_ph: tf.Tensor
        :return: (tf.Tensor, tf.Tensor)
        """
        raise NotImplementedError


@six.add_metaclass(abc.ABCMeta)
class ClassifierDifferentiable(ClassifierWithLogits):
    def __init__(self, x_min, x_max, x_shape, x_dtype, y_dtype, n_class):
        ClassifierWithLogits.__init__(self,
                                      x_min=x_min,
                                      x_max=x_max,
                                      x_shape=x_shape,
                                      x_dtype=x_dtype,
                                      y_dtype=y_dtype,
                                      n_class=n_class)
