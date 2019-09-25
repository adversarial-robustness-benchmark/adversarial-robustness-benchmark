import tensorflow as tf
import numpy as np
import argparse

from realsafe.cifar10.ResNet_PGD_AT import ResNet_PGD_AT
from realsafe.cifar10.wideresnet_trades import WideResNet_TRADES

from realsafe.attack.bim import BIM
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

ITERATION = 100
MAGNITUDE = 8.0
ALPHA = 2.0

CONFIGS = {
    'bim': {},
    'mim': {
        'decay_factor': 1.0,
    },
}
ATTACKS = {
    'bim': BIM,
    'mim': MIM,
}

def distance_l_2(a, b):
    d = np.reshape(a - b, (a.shape[0], -1))
    return np.sqrt((d ** 2).sum(axis=1))


def distance_l_inf(a, b):
    d = np.reshape(a - b, (a.shape[0], -1))
    return np.abs(d).max(axis=1)

def source_to_target(xs):
    xs = xs.copy().astype(model_t.x_dtype.as_numpy_dtype)
    k = (model_t.x_max - model_t.x_min) / (model_s.x_max - model_s.x_min)
    b = model_t.x_min - k * model_s.x_min
    return xs * k + b

class IterationBenchmark(object):
    def __init__(self, method, basic_configs):
        RUNS = {
            'bim': self._run_basic,
            'mim': self._run_basic,
        }

        self.method = method
        self.basic_configs = basic_configs
        self.config = CONFIGS[self.method]
        self.goal = basic_configs['goal']

        with graph_s.as_default():
            self.attack = ATTACKS[self.method](model_s, 1)

        if basic_configs['distance_metric'] == 'l_2':
            self._distance = distance_l_2
        else:
            self._distance = distance_l_inf

        self._run = RUNS[self.method]

    def _run_basic(self, xs, ys, ys_target):
        self.attack.config(
            iteration=ITERATION,
            magnitude=MAGNITUDE,
            alpha=ALPHA,
            **self.basic_configs,
            **self.config
        )

        rs = dict()

        for i, xs_adv in enumerate(
                self.attack.batch_attack_iterator(xs, ys, ys_target, session_s)):
            print(" {}".format(i + 1))
            lb = session_t.run(labels_t, feed_dict={xs_ph_t: source_to_target(xs_adv)})[0]
            di = self._distance(source_to_target(xs_adv), source_to_target(xs))[0]
            rs[i + 1] = (lb, di)

        return rs

    def run(self, xs, ys, ys_target):
        rs = []

        for lo in range(len(xs)):
            print(lo)
            s = slice(lo, lo + 1)
            rs.append(self._run(xs[s], ys[s], ys_target[s]))

        return rs

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

    benchmark = IterationBenchmark(method, basic_configs) 
    rs = benchmark.run(xs, ys, ys_target)

    np.save(output, rs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--method', type=str, required=True,
        choices=['bim', 'mim'],
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
