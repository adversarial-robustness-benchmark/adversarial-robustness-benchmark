import tensorflow as tf
import numpy as np
import argparse

from realsafe.cifar10.ResNet_PGD_AT import ResNet_PGD_AT
from realsafe.cifar10.wideresnet_trades import WideResNet_TRADES

from realsafe.attack.bim import BIM
from realsafe.attack.fgsm import FGSM
from realsafe.attack.mim import MIM


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

graph_s = tf.Graph()
with graph_s.as_default():
    model_s = ResNet_PGD_AT()
    session_s = tf.Session(config=config)
    model_s.load(session=session_s)

graph_t = tf.Graph()
with graph_t.as_default():
    model_t = WideResNet_TRADES()
    session_t = tf.Session(config=config)
    model_t.load(session=session_t)
    xs_ph_t = tf.placeholder(tf.float32, (None,) + model_t.x_shape)
    ys_ph_t = tf.placeholder(tf.int32, (None,))

    logits_t, labels_t = model_t.logits_and_labels(xs_ph_t)


ITERATION = 10
INIT_DISTORTION = 0.1 * (model_s.x_max - model_s.x_min)
BIN_SEARCH_STEPS = 10

CONFIGS = {
    'fgsm': {},
    'bim': {
        'iteration': ITERATION,
    },
    'mim': {
        'iteration': ITERATION,
        'decay_factor': 1.0,
    },
}
ATTACKS = {
    'fgsm': FGSM,
    'bim': BIM,
    'mim': MIM,
}

def source_to_target(xs):
    xs = xs.copy().astype(model_t.x_dtype.as_numpy_dtype)
    k = (model_t.x_max - model_t.x_min) / (model_s.x_max - model_s.x_min)
    b = model_t.x_min - k * model_s.x_min
    return xs * k + b

class SuccessRateBenchmark(object):
    def __init__(self, method, basic_configs):
        RUNS = {
            'fgsm': SuccessRateBenchmark._bin_search_basic,
            'bim': SuccessRateBenchmark._bin_search_alpha,
            'mim': SuccessRateBenchmark._bin_search_alpha,
        }

        self.config = CONFIGS[method]
        self.basic_configs = basic_configs
        self.goal = basic_configs['goal']
        self.method = method

        with graph_s.as_default():
            self.attack = ATTACKS[method](model_s, 1)
        self._run = RUNS[method]

    def run(self, xs, ys, ys_target):
        xs_adv = np.zeros_like(xs)

        for lo in range(len(xs)):
            print(lo)
            s = slice(lo, lo + 1)
            xs_adv[s] = self._run(self, xs[s], ys[s], ys_target[s])

        return xs_adv

    def _bin_search_basic(self, xs, ys, ys_target):
        found = np.array([False])
        hi = np.zeros(shape=1, dtype=np.float32) + INIT_DISTORTION
        lo = np.zeros(shape=1, dtype=np.float32)
        xs_adv = np.zeros_like(xs)

        for i in range(10):
            print(i, hi, lo)
            self.attack.config(magnitude=hi,
                               **self.basic_configs, **self.config)
            xs_adv_ = self.attack.batch_attack(xs, ys, ys_target, session_s)
            ys_adv_ = session_t.run(labels_t, feed_dict={xs_ph_t: source_to_target(xs_adv_)})
            flag = ys_adv_ == ys_target if self.goal == 't' else ys_adv_ != ys
            cond = np.logical_and(np.logical_not(found), flag)
            xs_adv[cond] = xs_adv_[cond]
            found[cond] = True
            lo[np.logical_not(found)] = hi[np.logical_not(found)]
            hi[np.logical_not(found)] *= 2.0
            if found.all():
                break

        for i in range(BIN_SEARCH_STEPS):
            mi = (lo + hi) / 2
            print(i, mi)
            self.attack.config(magnitude=mi,
                               **self.basic_configs, **self.config)
            xs_adv_ = self.attack.batch_attack(xs, ys, ys_target, session_s)
            ys_adv_ = session_t.run(labels_t, feed_dict={xs_ph_t: source_to_target(xs_adv_)})
            succ = ys_adv_ == ys_target if self.goal == 't' else ys_adv_ != ys
            not_succ = np.logical_not(succ)
            hi[succ] = mi[succ]
            lo[not_succ] = mi[not_succ]
            xs_adv[succ] = xs_adv_[succ]

        return xs_adv

    def _bin_search_alpha(self, xs, ys, ys_target):
        found = np.array([False])
        hi = np.zeros(shape=1, dtype=np.float32) + INIT_DISTORTION
        lo = np.zeros(shape=1, dtype=np.float32)
        xs_adv = np.zeros_like(xs)

        for i in range(10):
            print(i, hi, lo)
            self.attack.config(
                magnitude=hi, alpha=hi * 1.5 / ITERATION,
                **self.basic_configs, **self.config
            )
            xs_adv_ = self.attack.batch_attack(xs, ys, ys_target, session_s)
            ys_adv_ = session_t.run(labels_t, feed_dict={xs_ph_t: source_to_target(xs_adv_)})
            flag = ys_adv_ == ys_target if self.goal == 't' else ys_adv_ != ys
            cond = np.logical_and(np.logical_not(found), flag)
            xs_adv[cond] = xs_adv_[cond]
            found[cond] = True
            lo[np.logical_not(found)] = hi[np.logical_not(found)]
            hi[np.logical_not(found)] *= 2.0
            if found.all():
                break

        for i in range(BIN_SEARCH_STEPS):
            mi = (lo + hi) / 2
            print(i, mi)
            self.attack.config(
                magnitude=mi, alpha=mi * 1.5 / ITERATION,
                **self.basic_configs, **self.config
            )
            xs_adv_ = self.attack.batch_attack(xs, ys, ys_target, session_s)
            ys_adv_ = session_t.run(labels_t, feed_dict={xs_ph_t: source_to_target(xs_adv_)})
            succ = ys_adv_ == ys_target if self.goal == 't' else ys_adv_ != ys
            not_succ = np.logical_not(succ)
            hi[succ] = mi[succ]
            lo[not_succ] = mi[not_succ]
            xs_adv[succ] = xs_adv_[succ]

        return xs_adv


def main(args):
    goal = args.goal
    distance = args.distance
    method = args.method
    xs = np.load(args.xs).astype(model_s.x_dtype.as_numpy_dtype)
    xs = (xs / 255.0) * (model_s.x_max - model_s.x_min) + model_s.x_min
    ys = np.load(args.ys).astype(model_s.y_dtype.as_numpy_dtype)
    ys_target = np.load(args.ys_target).astype(model_s.y_dtype.as_numpy_dtype)
    output = args.output

    basic_configs = {
        'goal': goal,
        'distance_metric': distance
    }

    benchmark = SuccessRateBenchmark(method, basic_configs)
    xs_adv = benchmark.run(xs, ys, ys_target)
    xs_adv = (xs_adv - model_s.x_min) / (model_s.x_max - model_s.x_min) * 255.0

    np.save(output, xs_adv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--method', type=str, required=True,
        choices=['fgsm', 'bim', 'mim'],
    )

    parser.add_argument(
        '--goal', type=str, required=True,
        choices=['t', 'tm', 'ut'],
        help='`t` stands for targeted, ' +
        '`tm` stands for targeted misclassification, ' +
        '`ut` stands for untargeted.'
    )

    parser.add_argument(
        '--distance', type=str, required=True,
        choices=['l_2', 'l_inf'],
    )

    parser.add_argument(
        '--xs', type=str, required=True,
        help='path to input image xs.npy'
    )

    parser.add_argument(
        '--ys', type=str, required=True,
        help='path to ys.npy'
    )

    parser.add_argument(
        '--ys-target', type=str, required=True,
        help='path to ys-target.npy'
    )

    parser.add_argument(
        '--output', type=str, required=True,
        help='path to output.npy'
    )

    args = parser.parse_args()
    main(args)
