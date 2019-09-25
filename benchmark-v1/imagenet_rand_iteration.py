import tensorflow as tf
import numpy as np
import argparse

from realsafe.dataset.ImageNet import load_batches_imagenet_test
from realsafe.dataset.ImageNet import load_image_of_class
from realsafe.ImageNet.randomization import Randomization_Inception_v3

from realsafe.attack.deepfool import DeepFool
from realsafe.attack.nes import NES
from realsafe.attack.spsa import SPSA
from realsafe.attack.nattack import NAttack
from realsafe.attack.boundary import Boundary
from realsafe.attack.evolutionary import Evolutionary


def main(args):
    goal = args.goal
    distance = args.distance
    method = args.method

    MODEL = Randomization_Inception_v3()
    ITERATION = 100
    LABEL_OFFSET = MODEL.n_class - 1000

    if distance == 'l_inf':
        # L_inf parameters
        MAGNITUDE = 16.0 / 255.0
        ALPHA = 2.0 / 255.0
        SPSA_LR = ALPHA
    else:
        # L_2 parameters
        MAGNITUDE = np.sqrt(1e-3 * MODEL.x_shape[0] * MODEL.x_shape[1] * MODEL.x_shape[2])
        ALPHA = MAGNITUDE * 0.15
        SPSA_LR = 0.01

    
    CONFIGS = {
        'deepfool': {},
        'nes': {
            'max_queries': 20000,
            'samples_per_draw': 100,
            'sigma': 1e-3 * (MODEL.x_max - MODEL.x_min),
            'learning_rate': ALPHA,
            'min_lr': ALPHA / 10,
            'lr_tuning': True,
            'plateau_length': 20,
        },
        'spsa': {
            'max_queries': 20000,
            'samples_per_draw': 100,
            'sigma': 1e-3 * (MODEL.x_max - MODEL.x_min),
            'learning_rate': SPSA_LR,
        },
        'nattack': {
            'max_queries': 20000,
            'samples_per_draw': 100,
            'sigma': 0.1,
            'learning_rate': 0.02,
        },
        'boundary': {
            'iteration': 10000,
            'max_directions': 25,
            'max_queries': 20000,
            'spherical_step': 1e-2,
            'source_step': 1e-2,
            'step_adaptation': 1.5,
            'logging': False,
        },
        'evolutionary': {
            'max_queries': 20000,
            'mu': 1e-2,
            'sigma': 3e-2,
            'sample_size': 32,
            'logging': False,
        },
    }
    ATTACKS = {
        'deepfool': DeepFool,
        'nes': NES,
        'spsa': SPSA,
        'nattack': NAttack,
        'boundary': Boundary,
        'evolutionary': Evolutionary,
    }

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    SESSION = tf.Session(config=config)

    MODEL.load(session=SESSION)
    XS_PH = tf.placeholder(MODEL.x_dtype, (None, *MODEL.x_shape))
    YS_PH = tf.placeholder(MODEL.y_dtype, (None,))
    LOGITS, LABELS = MODEL.logits_and_labels(XS_PH)

    def distance_l_2(a, b):
        d = np.reshape(a - b, (a.shape[0], -1))
        return np.sqrt((d ** 2).sum(axis=1))

    def distance_l_inf(a, b):
        d = np.reshape(a - b, (a.shape[0], -1))
        return np.abs(d).max(axis=1)

    class IterationBenchmark(object):
        def __init__(self, method, basic_configs):
            RUNS = {
                'deepfool': self._run_deepfool,
                'nes': self._run_score_based,
                'spsa': self._run_score_based,
                'nattack': self._run_score_based,
                'boundary': self._run_decision_based,
                'evolutionary': self._run_decision_based,
            }

            self.method = method
            self.basic_configs = basic_configs
            self.config = CONFIGS[self.method]
            self.attack = ATTACKS[self.method](MODEL, 1)
            self.goal = basic_configs['goal']

            if basic_configs['distance_metric'] == 'l_2':
                self._distance = distance_l_2
            else:
                self._distance = distance_l_inf

            self._run = RUNS[self.method]

        def _run_deepfool(self, xs, ys, ys_target):
            self.attack.config(
                iteration=ITERATION,
                **self.basic_configs,
                **self.config
            )

            rs = dict()

            for i, xs_adv in enumerate(self.attack.batch_attack_iterator(
                    xs, ys, ys_target, SESSION)):
                print(" {}".format(i + 1))
                lb = SESSION.run(LABELS, feed_dict={XS_PH: xs_adv})[0]
                di = self._distance(xs_adv, xs)[0]
                rs[i + 1] = (lb, di)

            return rs, xs_adv

        def _run_score_based(self, xs, ys, ys_target):
            self.attack.config(
                magnitude=MAGNITUDE,
                **self.basic_configs,
                **self.config
            )
            xs_adv, q = self.attack.batch_attack(xs, ys, ys_target, SESSION)
            print(ys, ys_target, SESSION.run(LABELS, feed_dict={XS_PH: xs})[
                  0], SESSION.run(LABELS, feed_dict={XS_PH: xs_adv})[0])
            print(np.max(xs_adv), np.min(xs_adv), np.max(
                xs_adv - xs), np.min(xs_adv - xs), q)
            lb = SESSION.run(LABELS, feed_dict={XS_PH: xs_adv})[0]

            return (lb, q)

        def _run_decision_based(self, xs, ys, ys_target):
            if self.goal == 'ut' or self.goal == 'tm':
                starting_point = np.random.uniform(
                    MODEL.x_min, MODEL.x_max, size=MODEL.x_shape).astype(
                        MODEL.x_dtype.as_numpy_dtype)
                while SESSION.run(LABELS, feed_dict={XS_PH: starting_point[np.newaxis]}) == ys:
                    starting_point = np.random.uniform(
                        MODEL.x_min, MODEL.x_max, size=MODEL.x_shape).astype(
                            MODEL.x_dtype.as_numpy_dtype)
            else:
                starting_point = xs_for_each_label[ys_target].squeeze(axis=0)

            self.attack.config(
                **self.basic_configs,
                **self.config)

            xs_adv, dis_per_query = self.attack.batch_attack(
                xs, ys, ys_target, SESSION, starting_point)
            print(ys, ys_target, SESSION.run(LABELS, feed_dict={XS_PH: xs})[
                  0], SESSION.run(LABELS, feed_dict={XS_PH: xs_adv})[0])
            print(np.linalg.norm(xs_adv - xs), dis_per_query[-1])

            return (xs_adv, dis_per_query)

        def run(self, start, end):
            rs = []

            for filenames, xs, ys, ys_target in load_batches_imagenet_test(batch_size=1,
                                                                           x_min=MODEL.x_min,
                                                                           x_max=MODEL.x_max,
                                                                           x_shape=MODEL.x_shape,
                                                                           x_dtype=MODEL.x_dtype,
                                                                           y_dtype=MODEL.y_dtype,
                                                                           start=start,
                                                                           end=end,
                                                                           label_offset=LABEL_OFFSET,
                                                                           return_target_class=True):
                print(filenames)
                rs.append(self._run(xs, ys, ys_target))

            return rs

    def prepare_xs_for_each_label():
        n_class = MODEL.n_class
        xs_for_each_label = np.zeros([n_class, *MODEL.x_shape]).astype(
            MODEL.x_dtype.as_numpy_dtype)
        for i in range(n_class):
            for xs in load_image_of_class(label=i,
                                          x_min=MODEL.x_min,
                                          x_max=MODEL.x_max,
                                          x_shape=MODEL.x_shape,
                                          x_dtype=MODEL.x_dtype,
                                          y_dtype=MODEL.y_dtype,
                                          label_offset=LABEL_OFFSET):

                lb = SESSION.run(LABELS, feed_dict={XS_PH: xs})[0]
                if lb == i:
                    xs_for_each_label[i] = xs
                    break

        return xs_for_each_label

    output = args.output
    basic_configs = {
        'goal': goal,
        'distance_metric': distance
    }

    if method == 'boundary' or method == 'evolutionary':
        xs_for_each_label = prepare_xs_for_each_label()

    benchmark = IterationBenchmark(method, basic_configs)
    rs = benchmark.run(args.start, args.end)

    np.save(output, rs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--method', type=str, required=True,
        choices=['deepfool', 'nes', 'spsa', 'nattack',
                 'boundary', 'evolutionary'],
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
        '--start', type=int, required=True,
        help='start image index'
    )

    parser.add_argument(
        '--end', type=int, required=True,
        help='end image index'
    )

    parser.add_argument(
        '--output', type=str, required=True,
        help='path to output.npy'
    )

    args = parser.parse_args()
    main(args)
