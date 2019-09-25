from realsafe.dataset.ImageNet import load_batches_imagenet_test
from realsafe.benchmark.distortion_benchmark import DistortionBenchmarkBuilder
from realsafe.benchmark.utils import imagenet_distortion_benchmark_parser
from realsafe.ImageNet.inception_v3_randmix import Inception_V3_RandMix
import tensorflow as tf
import numpy as np
import os

BATCH_SIZE = 5

SESSION = tf.Session()

args = imagenet_distortion_benchmark_parser()

MODEL = Inception_V3_RandMix()
INIT_DISTORTION_L_2 = 32.0 * np.sqrt(1e-3 * np.prod(MODEL.x_shape))
INIT_DISTORTION_L_2 *= MODEL.x_max - MODEL.x_min
INIT_DISTORTION_L_INF = 1.0 * (MODEL.x_max - MODEL.x_min)
LABEL_OFFSET = MODEL.n_class - 1000

MODEL.load(session=SESSION)
ITERATION = 20

builder = DistortionBenchmarkBuilder()
builder.init_distortion_l_2(INIT_DISTORTION_L_2)
builder.init_distortion_l_inf(INIT_DISTORTION_L_INF)
builder.search_steps(0)
builder.binsearch_steps(15)
builder.batch_size(BATCH_SIZE)
builder.no_batch_pred(True)

builder.config_init_l_2('bim', {})
builder.config_l_2('bim', {
    'iteration': ITERATION,
    'session': SESSION
})

builder.config_init_l_inf('bim', {})
builder.config_l_inf('bim', {
    'iteration': ITERATION,
    'session': SESSION
})


builder.config_init_l_2('mim', {
    'decay_factor': 1.0
})
builder.config_l_2('mim', {
    'iteration': ITERATION,
    'session': SESSION
})

builder.config_init_l_inf('mim', {
    'decay_factor': 1.0
})
builder.config_l_inf('mim', {
    'iteration': ITERATION,
    'session': SESSION
})


builder.config_init_l_2('fgsm', {})
builder.config_l_2('fgsm', {
    'session': SESSION
})
builder.config_init_l_inf('fgsm', {})
builder.config_l_inf('fgsm', {
    'session': SESSION
})

benchmark = \
    builder.build(SESSION, MODEL, args.method, args.goal, args.distance_metric)

xs_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, *MODEL.x_shape))
_, labels = MODEL.logits_and_labels(xs_ph)


def get_labels(xs_adv):
    rs = []
    for i in range(0, len(xs_adv), BATCH_SIZE):
        rs.append(SESSION.run(labels, feed_dict={xs_ph: xs_adv[i:i + BATCH_SIZE]}))
    return np.concatenate(rs)


os.makedirs(args.output_dir, exist_ok=True)

for count, (filenames, xs, ys, ys_target) in enumerate(
    load_batches_imagenet_test(
        batch_size=BATCH_SIZE, x_min=MODEL.x_min, x_max=MODEL.x_max,
        x_shape=MODEL.x_shape, x_dtype=MODEL.x_dtype, y_dtype=MODEL.y_dtype,
        start=0, end=1000, label_offset=LABEL_OFFSET,
        return_target_class=True)):
    output_filename = os.path.join(args.output_dir, '%d_rs.npy' % count)

    xs_adv = benchmark.run(xs, ys, ys_target)
    ys_adv = get_labels(xs_adv)
    np.save(output_filename, {'xs': xs_adv, 'ys': ys_adv})
    print('ys        ', (ys_adv == ys).astype(np.float32).mean())
    print('ys_target ', (ys_adv == ys_target).astype(np.float32).mean())
