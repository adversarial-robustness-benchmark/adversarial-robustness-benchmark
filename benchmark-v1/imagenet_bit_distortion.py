import tensorflow as tf
import numpy as np
import argparse

from realsafe.dataset.ImageNet import load_batches_imagenet_test
from realsafe.ImageNet.inception_v3 import Inception_V3
from realsafe.defense.bit_depth_reduction import bit_depth_reduction_classifier_differentiable

from realsafe.attack.nes import NES
from realsafe.attack.spsa import SPSA
from realsafe.attack.nattack import NAttack


def main(args):
    goal = args.goal
    distance = args.distance
    method = args.method

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    SESSION = tf.Session(config=config)
    BIT_SESSION = tf.Session(config=config)

    paras = (0.0, 1.0, (299, 299, 3), tf.float32, tf.int32, 1001, 4, BIT_SESSION)
    IncV3_bit = bit_depth_reduction_classifier_differentiable(*paras)(Inception_V3)

    MODEL = IncV3_bit()
    BIN_SEARCH_STEPS = 10
    LABEL_OFFSET = MODEL.n_class - 1000

    if distance == 'l_inf':
        # L_inf parameters
        INIT_DISTORTION = 0.1 * (MODEL.x_max - MODEL.x_min)
        SPSA_LR_MUL = 0.15
    else:
        # L_2 parameters
        INIT_DISTORTION = np.sqrt(1e-3 * MODEL.x_shape[0] * MODEL.x_shape[1] * MODEL.x_shape[2]) * (MODEL.x_max - MODEL.x_min)
        SPSA_LR_MUL = 0.01 / INIT_DISTORTION

    CONFIGS = {
        'nes': {
            'max_queries': 20000,
            'samples_per_draw': 100,
            'sigma': 1e-3 * (MODEL.x_max - MODEL.x_min),
            'lr_tuning': True,
            'plateau_length': 20,
            'return_details': False,
        },
        'spsa': {
            'max_queries': 20000,
            'samples_per_draw': 100,
            'sigma': 1e-3 * (MODEL.x_max - MODEL.x_min),
            'return_details': False,
        },
        'nattack': {
            'max_queries': 20000,
            'samples_per_draw': 100,
            'sigma': 0.1,
            'learning_rate': 0.02,
            'return_details': False,
        },
    }
    ATTACKS = {
        'nes': NES,
        'spsa': SPSA,
        'nattack': NAttack,
    }

    MODEL.load(session=SESSION)
    XS_PH = tf.placeholder(MODEL.x_dtype, (None, *MODEL.x_shape))
    YS_PH = tf.placeholder(MODEL.y_dtype, (None,))
    LOGITS, LABELS = MODEL.logits_and_labels(XS_PH)

    class SuccessRateBenchmark(object):
        def __init__(self, method, basic_configs):
            RUNS = {
                'nes': SuccessRateBenchmark._bin_search_nes,
                'spsa': SuccessRateBenchmark._bin_search_nes,
                'nattack': SuccessRateBenchmark._bin_search_nes,
            }

            self.config = CONFIGS[method]
            self.basic_configs = basic_configs
            self.goal = basic_configs['goal']
            self.method = method

            self.attack = ATTACKS[method](MODEL, 1)
            self._run = RUNS[method]

            self._optimized_configured = False

        def _bin_search_nes(self, xs, ys, ys_target):
            found = np.array([False])
            hi = np.zeros(shape=1, dtype=np.float32) + INIT_DISTORTION
            lo = np.zeros(shape=1, dtype=np.float32)
            xs_adv = np.zeros_like(xs)

            for i in range(10):
                print(i, hi, lo)
                if self.method == 'nes':
                    specific_config = {'magnitude': hi,
                                       'learning_rate': hi * 0.15,
                                       'min_lr': hi * 0.015
                                       }
                elif self.method == 'spsa':
                    specific_config = {'magnitude': hi,
                                       'learning_rate': hi * SPSA_LR_MUL}
                elif self.method == 'nattack':
                    specific_config = {'magnitude': hi}

                self.attack.config(
                    **specific_config, **self.basic_configs, **self.config
                )
                xs_adv_ = self.attack.batch_attack(xs, ys, ys_target, SESSION)
                ys_adv_ = SESSION.run(LABELS, feed_dict={XS_PH: xs_adv_})
                flag = ys_adv_ == ys_target if self.goal == 't' else ys_adv_ != ys
                cond = np.logical_and(np.logical_not(found), flag)
                xs_adv[cond] = xs_adv_[cond]
                found[cond] = True
                lo[np.logical_not(found)] = hi[np.logical_not(found)]
                hi[np.logical_not(found)] *= 2.0

                if found.all():
                    break
            else:
                return xs_adv

            for i in range(BIN_SEARCH_STEPS):
                mi = (lo + hi) / 2
                print(i, mi)
                if self.method == 'nes':
                    specific_config = {'magnitude': mi,
                                       'learning_rate': mi * 0.15,
                                       'min_lr': mi * 0.015}
                elif self.method == 'spsa':
                    specific_config = {'magnitude': mi,
                                       'learning_rate': mi * SPSA_LR_MUL}
                elif self.method == 'nattack':
                    specific_config = {'magnitude': mi}
                self.attack.config(
                    **specific_config, **self.basic_configs, **self.config
                )
                xs_adv_ = self.attack.batch_attack(xs, ys, ys_target, SESSION)
                ys_adv_ = SESSION.run(LABELS, feed_dict={XS_PH: xs_adv_})
                succ = ys_adv_ == ys_target if self.goal == 't' else ys_adv_ != ys
                not_succ = np.logical_not(succ)
                hi[succ] = mi[succ]
                lo[not_succ] = mi[not_succ]
                xs_adv[succ] = xs_adv_[succ]

            return xs_adv

        def run(self, start, end):
            xs_adv = np.zeros((end - start,) + MODEL.x_shape)
            idx = 0

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
                xs_adv[idx] = self._run(self, xs, ys, ys_target)
                idx += 1

            return xs_adv

    output = args.output
    basic_configs = {
        'goal': goal,
        'distance_metric': distance
    }

    benchmark = SuccessRateBenchmark(method, basic_configs)
    xs_adv = benchmark.run(args.start, args.end)

    np.save(output, xs_adv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--method', type=str, required=True,
        choices=['nes', 'spsa', 'nattack'],
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
