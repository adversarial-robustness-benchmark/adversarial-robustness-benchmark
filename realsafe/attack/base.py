from __future__ import absolute_import

import six
import abc


@six.add_metaclass(abc.ABCMeta)
class Attack(object):
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    @abc.abstractmethod
    def config(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def batch_attack(self, xs, ys, ys_target, session):
        """
        :param xs: np.ndarray
        :param ys: np.ndarray
        :param session: tf.Session
        :return: np.ndarray
        """
        raise NotImplementedError


@six.add_metaclass(abc.ABCMeta)
class IterativeMethod(object):
    @abc.abstractmethod
    def batch_attack_iterator(self, xs, ys, ys_target, session):
        raise NotImplementedError


@six.add_metaclass(abc.ABCMeta)
class OptimizedMethod(object):
    pass


@six.add_metaclass(abc.ABCMeta)
class ConstrainedMethod(object):
    pass
