import tensorflow as tf
import numpy as np

from realsafe.attack.base import Attack, IterativeMethod
from realsafe.model import ClassifierDifferentiable


class DIM(Attack):
    """
    l_2, l_inf
    constrained
    """

    def __init__(self, model, batch_size, width_min=None, prob=0.9):
        assert isinstance(model, ClassifierDifferentiable)
        assert len(model.x_shape) == 3
        Attack.__init__(self, model, batch_size)

        self.width_max = self.model.x_shape[1]
        self.width_min = width_min if width_min else self.width_max // 2
        self.prob = prob

        self.x_dim = 1
        for x in self.model.x_shape:
            self.x_dim *= x

        self.batch_size = batch_size
        self.xs_ph = tf.placeholder(model.x_dtype, (batch_size, self.x_dim))
        self.ys_ph = tf.placeholder(model.y_dtype, (batch_size,))

        self.xs_var = tf.Variable(tf.zeros_like(self.xs_ph))
        self.xs_adv = tf.Variable(tf.zeros_like(self.xs_ph))
        self.ys_var = tf.Variable(tf.zeros_like(self.ys_ph))
        self.g_var = tf.Variable(tf.zeros_like(self.xs_var))

        xs_adv_di = self._input_diversity(
            tf.reshape(self.xs_adv, (self.batch_size, *self.model.x_shape)))
        logits, _ = self.model.logits_and_labels(xs_adv_di)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.ys_var, logits=logits)
        self.grad = tf.gradients(loss, self.xs_adv)[0]
        self.grad_l1 = tf.reduce_sum(tf.abs(self.grad), axis=1, keepdims=True)
        self.setup = [self.xs_var.assign(self.xs_ph),
                      self.xs_adv.assign(self.xs_ph),
                      self.ys_var.assign(self.ys_ph),
                      tf.variables_initializer([self.g_var])]

    def config(self, **kwargs):
        self.goal = kwargs["goal"]
        self.magnitude = np.array(kwargs["magnitude"]).reshape((-1, 1))
        self.alpha = np.array(kwargs["alpha"]).reshape((-1, 1))
        self.iteration = kwargs["iteration"]
        self.decay_factor = kwargs["decay_factor"]
        self.distance_metric = kwargs["distance_metric"]

        self.update_g_step = self.g_var.assign(
            self.decay_factor * self.g_var + self.grad / self.grad_l1)

        if self.goal == "t" or self.goal == "tm":
            g = -self.g_var
        elif self.goal == "ut":
            g = self.g_var
        else:
            raise NotImplementedError

        if self.distance_metric == "l_2":
            g_norm = tf.maximum(1e-12, tf.norm(g, axis=1, keepdims=True))
            g_unit = g / g_norm
            delta = tf.clip_by_norm(self.xs_adv + self.alpha * g_unit - self.xs_var,
                                    self.magnitude, axes=1)
            self.update_xs_step = self.xs_adv.assign(
                tf.clip_by_value(self.xs_var + delta, self.model.x_min,
                                 self.model.x_max))
        elif self.distance_metric == "l_inf":
            delta = tf.clip_by_value(
                self.xs_adv + self.alpha *
                tf.sign(g) - self.xs_var, -self.magnitude,
                self.magnitude)
            self.update_xs_step = self.xs_adv.assign(
                tf.clip_by_value(self.xs_var + delta, self.model.x_min,
                                 self.model.x_max))
        else:
            raise NotImplementedError

    def batch_attack(self, xs, ys, ys_target, session):
        assert xs.shape[0] == self.batch_size
        label = ys if self.goal == "ut" else ys_target
        assert label.shape[0] == self.batch_size

        session.run(self.setup, feed_dict={
            self.xs_ph: xs.reshape((self.batch_size, -1)),
            self.ys_ph: label
        })

        for _ in range(self.iteration):
            session.run(self.update_g_step)
            session.run(self.update_xs_step)

        xs_shape = (self.batch_size,) + self.model.x_shape
        return session.run(tf.reshape(self.xs_adv, xs_shape))

    def batch_attack_iterator(self, xs, ys, ys_target, session):
        assert xs.shape[0] == self.batch_size
        label = ys if self.goal == "ut" else ys_target
        assert ys.shape[0] == self.batch_size

        session.run(self.setup, feed_dict={
            self.xs_ph: xs.reshape((self.batch_size, -1)),
            self.ys_ph: label
        })

        xs_shape = (self.batch_size,) + self.model.x_shape
        xs_adv = tf.reshape(self.xs_adv, xs_shape)
        for _ in range(self.iteration):
            session.run(self.update_g_step)
            session.run(self.update_xs_step)
            yield session.run(xs_adv)

    def _input_diversity(self, xs):
        # xs shall be of shape (batch_size, width, width, channel)
        width = tf.random_uniform(
            (), self.width_min, self.width_max, dtype=tf.int32)
        xs_resized = tf.image.resize_nearest_neighbor(xs, [width, width],
                                                      align_corners=True)
        remain = self.width_max - width
        pad_top = tf.random_uniform((), 0, remain, dtype=tf.int32)
        pad_bottom = remain - pad_top
        pad_left = tf.random_uniform((), 0, remain, dtype=tf.int32)
        pad_right = remain - pad_left
        xs_padded = tf.pad(xs_resized,
                           [[0, 0], [pad_top, pad_bottom], [
                               pad_left, pad_right], [0, 0]],
                           constant_values=self.model.x_min)
        xs_padded.set_shape((xs.shape[0],
                             self.width_max, self.width_max, xs.shape[3]))
        return tf.cond(
            tf.random_uniform((), minval=0.0, maxval=1.0) < self.prob,
            lambda: xs_padded,
            lambda: xs)
