from __future__ import absolute_import, division
import tensorflow as tf

from realsafe.attack.base import Attack, IterativeMethod
from realsafe.model import ClassifierDifferentiable


class DeepFool(Attack, IterativeMethod):
    """
    l_2
    optimized
    """

    def __init__(self, model, batch_size, overshot=0.02):
        Attack.__init__(self, model=model, batch_size=batch_size)
        assert isinstance(self.model, ClassifierDifferentiable)

        self.batch_size = batch_size
        self.overshot = overshot

        x_dim = 1
        for x in self.model.x_shape:
            x_dim *= x

        self.x_dim = x_dim

        self.xs_ph = tf.placeholder(self.model.x_dtype,
                                    shape=(batch_size, x_dim))
        self.ys_ph = tf.placeholder(self.model.y_dtype, shape=(batch_size,))
        self.xs_var = tf.Variable(tf.zeros_like(self.xs_ph))
        self.ys_var = tf.Variable(tf.zeros_like(self.ys_ph))
        self.xs_adv = tf.Variable(tf.zeros_like(self.xs_ph))

        logits, self.labels = self.model.logits_and_labels(
            xs_ph=tf.reshape(self.xs_adv, (batch_size,) + self.model.x_shape))

        k0s = tf.stack((tf.range(self.batch_size), self.ys_var), axis=1)

        grads = [tf.gradients(logits[:, i], self.xs_adv)[0]
                 for i in range(self.model.n_class)]
        grads = tf.stack(grads, axis=0)
        grads = tf.transpose(grads, (1, 0, 2))

        yk0s = tf.expand_dims(tf.gather_nd(logits, k0s), axis=1)
        gradk0s = tf.expand_dims(tf.gather_nd(grads, k0s), axis=1)

        self.fs = tf.abs(yk0s - logits)
        self.ws = grads - gradk0s

        self.iteration = None
        self.distance_metric = None

    def _config_l_2(self):
        ws_norm = tf.norm(tf.reshape(
            self.ws, (-1, self.model.n_class, self.x_dim)), axis=-1)

        # for index = k0, ws_norm = 0.0, fs = 0.0, ls = 0.0 / 0.0 = NaN, and
        # tf.argmin would ignore NaN
        ls = self.fs / ws_norm
        ks = tf.argmin(ls, axis=1, output_type=self.model.y_dtype)
        ks = tf.stack((tf.range(self.batch_size), ks), axis=1)

        fsks = tf.gather_nd(self.fs, ks)
        ws_normks = tf.gather_nd(ws_norm, ks)
        wsks = tf.gather_nd(self.ws, ks)
        rs = tf.reshape(fsks / tf.square(ws_normks),
                        (self.batch_size, 1)) * wsks

        eqs = tf.equal(self.labels, self.ys_var)
        self.flag = tf.reduce_any(eqs)
        flags = tf.reshape(
            tf.cast(eqs, self.model.x_dtype) * (1 + self.overshot),
            (self.batch_size, 1))
        xs_adv_next = self.xs_adv + flags * rs

        self.update_xs_adv_step = self.xs_adv.assign(
            tf.clip_by_value(xs_adv_next, self.model.x_min, self.model.x_max))

        self.setup = [self.xs_var.assign(self.xs_ph),
                      self.ys_var.assign(self.ys_ph),
                      self.xs_adv.assign(self.xs_ph)]

    def _config_l_inf(self):
        ws_norm = tf.reduce_sum(tf.abs(tf.reshape(
            self.ws, (-1, self.model.n_class, self.x_dim))), axis=-1)

        # for index = k0, ws_norm = 0.0, fs = 0.0, ls = 0.0 / 0.0 = NaN, and
        # tf.argmin would ignore NaN
        ls = self.fs / ws_norm
        ks = tf.argmin(ls, axis=1, output_type=self.model.y_dtype)
        ks = tf.stack((tf.range(self.batch_size), ks), axis=1)

        fsks = tf.gather_nd(self.fs, ks)
        ws_normks = tf.gather_nd(ws_norm, ks)
        ws_sign_ks = tf.gather_nd(tf.sign(self.ws), ks)
        rs = tf.reshape(fsks / ws_normks, (self.batch_size, 1)) * ws_sign_ks

        eqs = tf.equal(self.labels, self.ys_var)
        self.flag = tf.reduce_any(eqs)
        flags = tf.reshape(
            tf.cast(eqs, self.model.x_dtype) * (1 + self.overshot),
            (self.batch_size, 1))
        xs_adv_next = self.xs_adv + flags * rs

        self.update_xs_adv_step = self.xs_adv.assign(
            tf.clip_by_value(xs_adv_next, self.model.x_min, self.model.x_max))

        self.setup = [self.xs_var.assign(self.xs_ph),
                      self.ys_var.assign(self.ys_ph),
                      self.xs_adv.assign(self.xs_ph)]

    def config(self, **kwargs):
        self.iteration = kwargs["iteration"]

        distance_metric = kwargs["distance_metric"]
        if self.distance_metric is None or \
                self.distance_metric != distance_metric:
            self.distance_metric = distance_metric
            if self.distance_metric == 'l_2':
                self._config_l_2()
            elif self.distance_metric == 'l_inf':
                self._config_l_inf()
            else:
                raise NotImplementedError

        if kwargs["goal"] != "ut":
            raise NotImplementedError

    def batch_attack(self, xs, ys, ys_target, session):
        session.run(self.setup, feed_dict={
            self.xs_ph: xs.reshape((-1, self.x_dim)), self.ys_ph: ys
        })

        for _ in range(self.iteration):
            session.run(self.update_xs_adv_step)
            flag = session.run(self.flag)
            if not flag:
                break

        return session.run(self.xs_adv).reshape(
            (self.batch_size,) + self.model.x_shape)

    def batch_attack_iterator(self, xs, ys, ys_target, session):
        session.run(self.setup, feed_dict={
            self.xs_ph: xs.reshape((-1, self.x_dim)), self.ys_ph: ys
        })

        xs_adv = tf.reshape(
            self.xs_adv, (self.batch_size, *self.model.x_shape))

        flag = True
        for _ in range(self.iteration):
            if flag:
                session.run(self.update_xs_adv_step)
                flag = session.run(self.flag)

            yield session.run(xs_adv)
