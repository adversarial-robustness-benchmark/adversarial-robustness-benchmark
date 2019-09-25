import tensorflow as tf
import numpy as np
import argparse

from realsafe.attack.fgsm import FGSM
from realsafe.attack.bim import BIM
from realsafe.attack.mim import MIM


class DistortionBenchmark(object):
    def __init__(self, session, batch_size, model, method, goal,
                 distance_metric, search_steps, binsearch_steps,
                 init_distortion, config_init, config, no_batch_pred=False):
        RUNS = {
            'fgsm': self._binsearch_basic,
            'bim': self._binsearch_alpha,
            'mim': self._binsearch_alpha,
        }
        ATTACKS = {
            'fgsm': FGSM,
            'bim': BIM,
            'mim': MIM,
        }

        self._init_distortion = init_distortion
        self._attack = ATTACKS[method](model, batch_size, goal=goal,
                                       distance_metric=distance_metric,
                                       **config_init)
        self._config = config
        self._session = session
        self._goal = goal
        self._run = RUNS[method]
        self._search_steps = search_steps
        self._binsearch_steps = binsearch_steps
        self._batch_size = batch_size

        if no_batch_pred:
            self._xs_ph = tf.placeholder(shape=(1, *model.x_shape),
                                         dtype=model.x_dtype)
            _, self._labels = model.logits_and_labels(self._xs_ph)
            self._get_labels = self._get_labels_one
        else:
            self._xs_ph = tf.placeholder(shape=(batch_size, *model.x_shape),
                                         dtype=model.x_dtype)
            _, self._labels = model.logits_and_labels(self._xs_ph)
            self._get_labels = self._get_labels_batch

    def _get_labels_batch(self, xs_adv):
        return self._session.run(self._labels, feed_dict={self._xs_ph: xs_adv})

    def _get_labels_one(self, xs_adv):
        rs = []
        for i in range(len(xs_adv)):
            rs.append(self._session.run(self._labels, feed_dict={
                self._xs_ph: xs_adv[i:i + 1]})[0])
        return np.array(rs)

    def run(self, xs, ys, ys_target):
        n = len(xs)
        rs = []
        for hi in range(self._batch_size, n + 1, self._batch_size):
            s = slice(hi - self._batch_size, hi)
            rs.append(self._run(xs[s], ys[s], ys_target[s]))
        return np.concatenate(rs)

    def _binsearch_basic(self, xs, ys, ys_target):
        lo = np.zeros(self._batch_size, dtype=np.float32)
        hi = lo + self._init_distortion
        xs_result = np.zeros_like(xs)

        steps = 5
        exp = 2 ** steps
        for i in range(exp):
            print('search {}'.format(i))
            magnitude = (1.0 - i / float(exp)) * self._init_distortion
            self._attack.config(magnitude=magnitude, **self._config)
            xs_adv = self._attack.batch_attack(
                xs, ys, ys_target, self._session)
            ys_adv = self._get_labels(xs_adv)
            cond = ys_adv == ys_target if self._goal == 't' else ys_adv != ys
            print(cond.astype(np.float32).mean())
            xs_result[cond] = xs_adv[cond]
            hi[cond] = magnitude

        lo = hi - self._init_distortion / float(exp)

        for i in range(self._binsearch_steps - steps):
            print('binsearch {}'.format(i))
            mi = (lo + hi) / 2
            self._attack.config(magnitude=mi, **self._config)
            xs_adv = self._attack.batch_attack(
                xs, ys, ys_target, self._session)
            ys_adv = self._get_labels(xs_adv)
            succ = ys_adv == ys_target if self._goal == 't' else ys_adv != ys
            print(succ.astype(np.float32).mean())
            not_succ = np.logical_not(succ)
            hi[succ] = mi[succ]
            lo[not_succ] = mi[not_succ]
            xs_result[succ] = xs_adv[succ]

        return xs_result

    def _binsearch_alpha(self, xs, ys, ys_target):
        found = np.repeat(False, self._batch_size)
        lo = np.zeros(self._batch_size, dtype=np.float32)
        hi = lo + self._init_distortion
        xs_result = np.zeros_like(xs)

        for i in range(self._search_steps):
            print('search {}'.format(i))
            self._attack.config(
                magnitude=hi, alpha=hi * 1.5 / self._config['iteration'],
                **self._config
            )
            xs_adv = self._attack.batch_attack(
                xs, ys, ys_target, self._session)
            ys_adv = self._get_labels(xs_adv)
            flag = ys_adv == ys_target if self._goal == 't' else ys_adv != ys
            cond = np.logical_and(np.logical_not(found), flag)
            xs_result[cond] = xs_adv[cond]
            found[cond] = True
            not_found = np.logical_not(found)
            lo[not_found] = hi[not_found]
            hi[not_found] *= 2.0
            if found.all():
                break

        for i in range(self._binsearch_steps):
            print('binsearch {}'.format(i))
            mi = (lo + hi) / 2
            self._attack.config(
                magnitude=mi, alpha=mi * 1.5 / self._config['iteration'],
                **self._config
            )
            xs_adv = self._attack.batch_attack(
                xs, ys, ys_target, self._session)
            ys_adv = self._get_labels(xs_adv)
            succ = ys_adv == ys_target if self._goal == 't' else ys_adv != ys
            print(succ.astype(np.float32).mean())
            not_succ = np.logical_not(succ)
            hi[succ] = mi[succ]
            lo[not_succ] = mi[not_succ]
            xs_result[succ] = xs_adv[succ]

        return xs_result


class DistortionBenchmarkBuilder(object):
    def __init__(self):
        self._config_init_l_2 = dict()
        self._config_l_2 = dict()
        self._config_init_l_inf = dict()
        self._config_l_inf = dict()
        self._no_batch_pred = False

    def config_init_l_2(self, method, config):
        self._config_init_l_2[method] = config

    def config_l_2(self, method, config):
        self._config_l_2[method] = config

    def config_init_l_inf(self, method, config):
        self._config_init_l_inf[method] = config

    def config_l_inf(self, method, config):
        self._config_l_inf[method] = config

    def search_steps(self, search_steps):
        self._search_steps = search_steps

    def binsearch_steps(self, binsearch_steps):
        self._binsearch_steps = binsearch_steps

    def init_distortion_l_2(self, init_distortion):
        self._init_distortion_l_2 = init_distortion

    def init_distortion_l_inf(self, init_distortion):
        self._init_distortion_l_inf = init_distortion

    def batch_size(self, batch_size):
        self._batch_size = batch_size

    def no_batch_pred(self, no_batch_pred):
        self._no_batch_pred = no_batch_pred

    def build(self, session, model, method, goal, distance_metric):
        if distance_metric == 'l_2':
            config_init = self._config_init_l_2[method]
            config = self._config_l_2[method]
            init_distortion = self._init_distortion_l_2
        elif distance_metric == 'l_inf':
            config_init = self._config_init_l_inf[method]
            config = self._config_l_inf[method]
            init_distortion = self._init_distortion_l_inf

        return DistortionBenchmark(
            session, self._batch_size, model, method, goal, distance_metric,
            self._search_steps, self._binsearch_steps, init_distortion,
            config_init, config, no_batch_pred=self._no_batch_pred)
