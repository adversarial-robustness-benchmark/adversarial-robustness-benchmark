from __future__ import absolute_import
import numpy as np
import tensorflow as tf

from realsafe.attack import Attack
from realsafe.model import ClassifierDifferentiable


class FGSM(Attack):
    '''
    l_2, l_inf
    constrained
    '''

    def __init__(self, model, batch_size, goal, distance_metric):
        assert isinstance(model, ClassifierDifferentiable)
        Attack.__init__(self, model=model, batch_size=batch_size)

        self._goal = goal

        self.xs_ph = tf.placeholder(self.model.x_dtype,
                                    (self.batch_size,) + self.model.x_shape)
        self.ys_ph = tf.placeholder(self.model.y_dtype, (self.batch_size,))
        self.eps_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.eps_var = tf.Variable(tf.zeros((self.batch_size,),
                                            dtype=self.model.x_dtype))

        eps = tf.expand_dims(self.eps_var, 1)

        logits, _ = self.model.logits_and_labels(self.xs_ph)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.ys_ph, logits=logits)

        if goal == 't' or goal == 'tm':
            grad = -tf.gradients(loss, self.xs_ph)[0]
        elif goal == 'ut':
            grad = tf.gradients(loss, self.xs_ph)[0]
        else:
            raise NotImplementedError

        if distance_metric == 'l_2':
            grad_2d = tf.reshape(grad, (self.batch_size, -1))
            grad_norm = tf.maximum(1e-12, tf.norm(grad_2d, axis=1))
            grad_unit = grad_2d / tf.expand_dims(grad_norm, 1)
            d = eps * grad_unit
            d = tf.reshape(d, (self.batch_size, *self.model.x_shape))
        elif distance_metric == 'l_inf':
            d = eps * tf.sign(tf.reshape(grad, (self.batch_size, -1)))
            d = tf.reshape(d, (self.batch_size, *self.model.x_shape))
        else:
            raise NotImplementedError

        self._xs_adv = tf.clip_by_value(self.xs_ph + d,
                                        self.model.x_min, self.model.x_max)
        self.config_setup = self.eps_var.assign(self.eps_ph)

    def config(self, **kwargs):
        magnitude = np.array(kwargs['magnitude'])

        if magnitude.shape == ():
            magnitude = np.repeat(magnitude, self.batch_size)

        session = kwargs['session']
        session.run(self.config_setup, feed_dict={self.eps_ph: magnitude})

    def batch_attack(self, xs, ys, ys_target, session):
        ls = ys if self._goal == 'ut' else ys_target

        return session.run(self._xs_adv, feed_dict={
            self.xs_ph: xs,
            self.ys_ph: ls
        })
