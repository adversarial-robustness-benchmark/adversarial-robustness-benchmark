import tensorflow as tf
import argparse


def new_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def imagenet_iteration_benchmark_parser():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--method', type=str, required=True,
        choices=['bim', 'mim', 'cw', 'deepfool', 'fgsm',
                 'nes', 'spsa', 'nattack',
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
        '--distance-metric', type=str, required=True,
        choices=['l_2', 'l_inf'],
    )

    parser.add_argument(
        '--output-dir', required=True,
        help='path to output'
    )

    return parser.parse_args()


def imagenet_distortion_benchmark_parser():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--method', type=str, required=True,
        choices=['bim', 'mim', 'cw', 'deepfool', 'fgsm',
                 'nes', 'spsa', 'nattack',
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
        '--distance-metric', type=str, required=True,
        choices=['l_2', 'l_inf'],
    )

    parser.add_argument(
        '--output-dir', required=True,
        help='path to output'
    )

    return parser.parse_args()


def iteartion_benchmark_parser():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--method', type=str, required=True,
        choices=['bim', 'mim', 'cw', 'deepfool',
                 'nes', 'spsa', 'nattack',
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
        '--distance-metric', type=str, required=True,
        choices=['l_2', 'l_inf'],
    )

    parser.add_argument(
        '--xs', required=True, action='append',
        help='path to input image xs.npy'
    )

    parser.add_argument(
        '--ys', required=True, action='append',
        help='path to ys.npy'
    )

    parser.add_argument(
        '--ys-target', required=True, action='append',
        help='path to ys-target.npy'
    )

    parser.add_argument(
        '--output', required=True, action='append',
        help='path to output.npy'
    )

    return parser.parse_args()


def distortion_benchmark_parser():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--method', type=str, required=True,
        choices=['bim', 'mim', 'cw', 'deepfool', 'fgsm',
                 'nes', 'spsa', 'nattack',
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
        '--distance-metric', type=str, required=True,
        choices=['l_2', 'l_inf'],
    )

    parser.add_argument(
        '--xs', required=True, action='append',
        help='path to input image xs.npy'
    )

    parser.add_argument(
        '--ys', required=True, action='append',
        help='path to ys.npy'
    )

    parser.add_argument(
        '--ys-target', required=True, action='append',
        help='path to ys-target.npy'
    )

    parser.add_argument(
        '--output', required=True, action='append',
        help='path to output.npy'
    )

    return parser.parse_args()
