import tensorflow as tf
import numpy as np

from realsafe.attack.bim import BIM
from realsafe.attack.mim import MIM
from realsafe.attack.cw import CW
from realsafe.attack.deepfool import DeepFool
from realsafe.attack.nes import NES
from realsafe.attack.spsa import SPSA
from realsafe.attack.nattack import NAttack
from realsafe.attack.boundary import Boundary
from realsafe.attack.evolutionary import Evolutionary


class IterationBenchmark(object):
    def __init__(self, batch_size, session, model, method, goal,
                 distance_metric, iteration, config_init, config,
                 no_batch_pred):
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

        def distance_l_2(a, b):
            d = np.reshape(a - b, (a.shape[0], -1))
            return np.sqrt((d ** 2).sum(axis=1))

        def distance_l_inf(a, b):
            d = np.reshape(a - b, (a.shape[0], -1))
            return np.abs(d).max(axis=1)

        DISTANCES = {
            'l_2': distance_l_2,
            'l_inf': distance_l_inf,
        }
        RUNS = {
            'cw': self._run_cw,
            'mim': self._run_basic,
            'bim': self._run_basic,
        }
        ATTACKS = {
            'bim': BIM,
            'mim': MIM,
            'cw': CW,
            'deepfool': DeepFool,
            'nes': NES,
            'spsa': SPSA,
            'nattack': NAttack,
            'boundary': Boundary,
            'evolutionary': Evolutionary,
        }

        self._attack = ATTACKS[method](model, batch_size, goal=goal,
                                       distance_metric=distance_metric,
                                       **config_init)
        self._distance = DISTANCES[distance_metric]
        self._iteration = iteration
        self._config = config
        self._session = session

        self.run = RUNS[method]

    def _get_labels_batch(self, xs_adv):
        return self._session.run(self._labels, feed_dict={self._xs_ph: xs_adv})

    def _get_labels_one(self, xs_adv):
        rs = []
        for i in range(len(xs_adv)):
            rs.append(self._session.run(self._labels, feed_dict={
                self._xs_ph: xs_adv[i:i + 1]})[0])
        return np.array(rs)

    def _run_basic(self, xs_batch, ys_batch, ys_target_batch):
        self._config['iteration'] = self._iteration
        self._attack.config(**self._config)

        n = len(xs_batch)
        rs = []
        for hi in range(self._batch_size, n + 1, self._batch_size):
            lo = hi - self._batch_size
            s = slice(lo, hi)
            xs, ys, ys_target = xs_batch[s], ys_batch[s], ys_target_batch[s]
            r = dict()
            for i, xs_adv in enumerate(self._attack.batch_attack_iterator(
                    xs, ys, ys_target, self._session)):
                print('{} {} iter = {}'.format(lo, hi, i))
                lb = self._get_labels(xs_adv)
                di = self._distance(xs_adv, xs)
                r[i + 1] = (lb, di)
            rs.append(r)
        return rs

    def _run_cw(self, xs_batch, ys_batch, ys_target_batch):
        n = len(xs_batch)
        rs = []
        for hi in range(self._batch_size, n + 1, self._batch_size):
            s = slice(hi - self._batch_size, hi)
            xs, ys, ys_target = xs_batch[s], ys_batch[s], ys_target_batch[s]
            r = dict()
            for i in range(10, self._iteration + 1, 10):
                print("{}".format(i))
                self._config['iteration'] = i
                self._attack.config(**self._config)
                xs_adv = self._attack.batch_attack(
                    xs, ys, ys_target, self._session)
                lb = self._get_labels(xs_adv)
                di = self._distance(xs_adv, xs)
                r[i] = (lb, di)
            rs.append((r, xs_adv))
        return rs


class IterationBenchmarkBuilder(object):
    def __init__(self):
        self._no_batch_pred = False
        self._config_init_l_2 = dict()
        self._config_l_2 = dict()
        self._config_init_l_inf = dict()
        self._config_l_inf = dict()

    def config_init_l_2(self, method, config):
        self._config_init_l_2[method] = config

    def config_l_2(self, method, config):
        self._config_l_2[method] = config

    def config_init_l_inf(self, method, config):
        self._config_init_l_inf[method] = config

    def config_l_inf(self, method, config):
        self._config_l_inf[method] = config

    def iteration(self, iteration):
        self._iteration = iteration

    def batch_size(self, batch_size):
        self._batch_size = batch_size

    def no_batch_pred(self, no_batch_pred):
        self._no_batch_pred = no_batch_pred

    def build(self, session, model, method, goal, distance_metric):
        if distance_metric == 'l_2':
            config_init = self._config_init_l_2[method]
            config = self._config_l_2[method]
        elif distance_metric == 'l_inf':
            config_init = self._config_init_l_inf[method]
            config = self._config_l_inf[method]

        return IterationBenchmark(
            self._batch_size, session, model, method, goal, distance_metric,
            self._iteration, config_init, config, self._no_batch_pred)
