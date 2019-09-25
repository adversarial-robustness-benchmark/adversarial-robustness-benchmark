import tensorflow as tf
import numpy as np

from realsafe.attack.base import Attack
from realsafe.attack.utils import get_xs_ph, get_ys_ph
from realsafe.model import ClassifierDifferentiable


class CW(Attack):
    '''
    l_2, l_inf
    optimized
    '''

    def __init__(self, model, batch_size, goal, distance_metric, learning_rate,
                 confidence):
        assert isinstance(model, ClassifierDifferentiable)
        Attack.__init__(self, model, batch_size)

        self.confidence = confidence

        def scale(vec, dst_lo, dst_hi, src_lo, src_hi):
            k = (dst_hi - dst_lo) / (src_hi - src_lo)
            b = dst_lo - k * src_lo
            return k * vec + b

        def scale_to_model(vec):
            return scale(vec, self.model.x_min, self.model.x_max, -1.0, 1.0)

        def scale_to_tanh(vec):
            return scale(vec, 1e-6 - 1, 1 - 1e-6,
                         self.model.x_min, self.model.x_max)

        model_xs_shape = (self.batch_size, *self.model.x_shape)

        xs_shape = (self.batch_size, np.prod(self.model.x_shape))

        xs_zeros = tf.zeros(xs_shape, dtype=self.model.x_dtype)

        self.xs_ph = get_xs_ph(self.model, self.batch_size)
        self.ys_ph = get_ys_ph(self.model, self.batch_size)
        self.cs_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))

        xs_var = tf.Variable(xs_zeros)
        ys_var = tf.Variable(tf.zeros_like(self.ys_ph))
        cs_var = tf.Variable(tf.zeros_like(self.cs_ph))

        d_ws = tf.Variable(xs_zeros)
        ws = tf.atanh(scale_to_tanh(xs_var)) + d_ws

        self.xs_adv = scale_to_model(tf.tanh(ws))
        self.xs_adv_output = tf.reshape(self.xs_adv, model_xs_shape)

        logits, _ = self.model.logits_and_labels(self.xs_adv_output)

        ys_one_hot = tf.one_hot(ys_var, self.model.n_class)

        logit_target = tf.reduce_sum(ys_one_hot * logits, 1)
        logit_other = (1 - ys_one_hot) * logits
        logit_other = logit_other - 0.5 * self.model.x_dtype.max * ys_one_hot
        logit_other = tf.reduce_max(logit_other, 1)

        self.setup_xs = xs_var.assign(tf.reshape(self.xs_ph, xs_shape))
        self.setup_ys = ys_var.assign(self.ys_ph)
        self.setup_cs = cs_var.assign(self.cs_ph)
        self.setup_d_ws = d_ws.assign(tf.zeros_like(d_ws))

        if distance_metric == 'l_2':
            dists = tf.reduce_sum(tf.square(self.xs_adv - xs_var), axis=1)
        elif distance_metric == 'l_inf':
            dists = tf.reduce_max(tf.abs(self.xs_adv - xs_var), axis=1)
        else:
            raise NotImplementedError

        if goal == 't' or goal == 'tm':
            score = tf.maximum(0.0, logit_other - logit_target + confidence)
        elif goal == 'ut':
            score = tf.maximum(0.0, logit_target - logit_other + confidence)
        else:
            raise NotImplementedError
        self.goal = goal

        loss = dists + cs_var * score

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.optimizer_step = optimizer.minimize(loss, var_list=[d_ws])
        self.setup_optimizer = tf.variables_initializer(optimizer.variables())

        self.score = score
        self.logits = logits
        self.dists = dists

    def config(self, **kwargs):
        cs = np.array(kwargs['cs'], dtype=self.model.x_dtype.as_numpy_dtype)
        self.cs = np.repeat(cs, self.batch_size) if cs.shape == () else cs
        self.iteration = kwargs['iteration']
        self.search_steps = kwargs['search_steps']
        self.binsearch_steps = kwargs['binsearch_steps']

    def batch_attack(self, xs, ys, ys_target, session):
        ys_flatten_max = self.batch_size * self.model.n_class
        ys_flatten = np.arange(0, ys_flatten_max, self.model.n_class) + ys

        cs = self.cs.copy()
        ys_input = ys_target if self.goal == 't' or self.goal == 'tm' else ys
        session.run((self.setup_xs, self.setup_ys, self.setup_d_ws),
                    feed_dict={self.xs_ph: xs, self.ys_ph: ys_input})

        xs_adv = np.copy(xs)

        # find c to begin with
        found = np.repeat(False, self.batch_size)
        min_dists = np.repeat(self.model.x_dtype.max, self.batch_size)
        for _ in range(self.search_steps):
            session.run(self.setup_optimizer)
            session.run(self.setup_cs, feed_dict={self.cs_ph: cs})
            for _ in range(self.iteration):
                session.run(self.optimizer_step)
                score_, logits_, xs_adv_, dists_ = session.run([
                    self.score, self.logits, self.xs_adv_output, self.dists])
                if self.goal == 'ut' or self.goal == 'tm':
                    diff = logits_.max(axis=1) - logits_.take(ys_flatten)
                    succ_ = diff > self.confidence
                else:
                    succ_ = score_ < 1e-12

                better_dists = dists_ < min_dists
                to_update = np.logical_and(succ_, better_dists)
                xs_adv[to_update] = xs_adv_[to_update]
                found[to_update] = True
            if np.all(found):
                break
            else:
                cs[np.logical_not(found)] *= 10.0

        cs_hi = cs
        cs_lo = np.zeros_like(cs)
        cs = (cs_hi + cs_lo) / 2

        # binsearch
        for _ in range(self.binsearch_steps):
            session.run(self.setup_optimizer)
            session.run(self.setup_cs, feed_dict={self.cs_ph: cs})

            succ = np.repeat(False, self.batch_size)
            for _ in range(self.iteration):
                session.run(self.optimizer_step)
                score_, logits_, xs_adv_, dists_ = session.run((
                    self.score, self.logits, self.xs_adv_output, self.dists))
                if self.goal == 'ut' or self.goal == 'tm':
                    succ_ = logits_.max(axis=1) - logits_.take(ys_flatten) \
                        > self.confidence
                else:
                    succ_ = score_ < 1e-12
                better_dists = dists_ < min_dists
                to_update = np.logical_and(succ_, better_dists)
                xs_adv[to_update] = xs_adv_[to_update]
                succ[to_update] = True

            cs_hi[succ] = cs[succ]
            not_succ = np.logical_not(succ)
            cs_lo[not_succ] = cs[not_succ]
            cs = (cs_hi + cs_lo) / 2.0

        return xs_adv
