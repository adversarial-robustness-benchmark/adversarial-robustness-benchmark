import numpy as np
import tensorflow as tf

from realsafe.attack.base import Attack, IterativeMethod
from realsafe.model import ClassifierDifferentiable
from realsafe.attack.utils import (get_xs_ph, get_ys_ph)


class BIM(Attack, IterativeMethod):
    '''
    l_2, l_inf
    constrained
    '''

    def __init__(self, model, batch_size, goal, distance_metric,
                 random_start=False):
        # TODO random_start
        assert isinstance(model, ClassifierDifferentiable)
        Attack.__init__(self, model=model, batch_size=batch_size)

        xs_shape = (self.batch_size, np.prod(self.model.x_shape))
        ys_shape = (self.batch_size,)
        model_xs_shape = (self.batch_size, *self.model.x_shape)
        xs_zeros = tf.zeros(xs_shape, dtype=self.model.x_dtype)

        self.xs_ph = get_xs_ph(self.model, self.batch_size)
        self.ys_ph = get_ys_ph(self.model, self.batch_size)
        self.eps_ph = tf.Variable(tf.zeros((self.batch_size,)))
        self.alpha_ph = tf.Variable(tf.zeros((self.batch_size,)))

        self.xs_var = tf.Variable(xs_zeros)
        self.ys_var = tf.Variable(tf.zeros(ys_shape, dtype=self.model.y_dtype))
        self.eps_var = tf.Variable(tf.zeros((batch_size,)))
        self.alpha_var = tf.Variable(tf.zeros((batch_size,)))

        self.xs_adv_var = tf.Variable(xs_zeros)
        self.xs_adv = tf.reshape(self.xs_adv_var, model_xs_shape)

        logits, _ = self.model.logits_and_labels(self.xs_adv)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.ys_var,
            logits=logits
        )

        self.grad = tf.gradients(loss, self.xs_adv_var)[0]
        self.config_setup = [
            self.eps_var.assign(self.eps_ph),
            self.alpha_var.assign(self.alpha_ph),
        ]
        xs_ph_in = tf.reshape(self.xs_ph, xs_shape)
        self.setup = [
            self.xs_var.assign(xs_ph_in),
            self.ys_var.assign(self.ys_ph),
            self.xs_adv_var.assign(xs_ph_in),
        ]

        eps = tf.expand_dims(self.eps_var, 1)
        alpha = tf.expand_dims(self.alpha_var, 1)

        if goal == 't' or goal == 'tm':
            grad = -self.grad
        elif goal == 'ut':
            grad = self.grad
        else:
            raise NotImplementedError

        if distance_metric == 'l_2':
            grad_norm = tf.maximum(1e-12, tf.norm(grad, axis=1))
            grad_unit = grad / tf.expand_dims(grad_norm, 1)
            xs_next = self.xs_adv_var - self.xs_var + alpha * grad_unit
            xs_next = self.xs_var + tf.clip_by_norm(xs_next, eps, axes=[1])
        elif distance_metric == 'l_inf':
            lo, hi = self.xs_var - eps, self.xs_var + eps
            xs_next = self.xs_adv_var + alpha * tf.sign(grad)
            xs_next = tf.clip_by_value(xs_next, lo, hi)
        else:
            raise NotImplementedError

        xs_next = tf.clip_by_value(xs_next, self.model.x_min, self.model.x_max)
        self.step = self.xs_adv_var.assign(xs_next)

        self.goal = goal

    def config(self, **kwargs):
        magnitude = np.array(kwargs['magnitude'])
        alpha = np.array(kwargs['alpha'])

        if magnitude.shape == ():
            magnitude = np.repeat(magnitude, self.batch_size)
        if alpha.shape == ():
            alpha = np.repeat(alpha, self.batch_size)

        self.iteration = kwargs['iteration']

        session = kwargs['session']
        session.run(self.config_setup, feed_dict={
            self.eps_ph: magnitude,
            self.alpha_ph: alpha,
        })

    def batch_attack(self, xs, ys, ys_target, session):
        ls = ys if self.goal == 'ut' else ys_target

        session.run(self.setup, feed_dict={
            self.xs_ph: xs,
            self.ys_ph: ls
        })
        for _ in range(self.iteration):
            session.run(self.step)
        return session.run(self.xs_adv)

    def batch_attack_iterator(self, xs, ys, ys_target, session):
        ls = ys if self.goal == 'ut' else ys_target

        session.run(self.setup, feed_dict={
            self.xs_ph: xs,
            self.ys_ph: ls
        })
        for _ in range(self.iteration):
            session.run(self.step)
            yield session.run(self.xs_adv)
