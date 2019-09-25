from realsafe.dataset.ImageNet import load_batches_imagenet_test
from realsafe.benchmark.utils import imagenet_iteration_benchmark_parser
from realsafe.benchmark.iteration_benchmark import IterationBenchmarkBuilder
from realsafe.ImageNet.inception_v3_randmix import Inception_V3_RandMix
import tensorflow as tf
import numpy as np
import os

BATCH_SIZE = 5

SESSION = tf.Session()

args = imagenet_iteration_benchmark_parser()

MODEL = Inception_V3_RandMix()
print('x_min={}\nx_max={}\nx_shape={}\nn_class={}'.format(
    MODEL.x_min, MODEL.x_max, MODEL.x_shape, MODEL.n_class))
LABEL_OFFSET = MODEL.n_class - 1000

ITERATION = 100

MAGNITUDE_L_INF = 16.0 / 255.0
ALPHA_L_INF = 2.0 / 255.0
SPSA_LR_L_INF = ALPHA_L_INF

MAGNITUDE_L_2 = np.sqrt(1e-3 * np.prod(MODEL.x_shape))
MAGNITUDE_L_2 *= MODEL.x_max - MODEL.x_min
ALPHA_L_2 = MAGNITUDE_L_2 * 0.15
SPSA_LR_L_2 = 0.01

MODEL.load(session=SESSION)

builder = IterationBenchmarkBuilder()

builder.config_init_l_inf('bim', {})
builder.config_l_inf('bim', {
    'magnitude': MAGNITUDE_L_INF,
    'alpha': ALPHA_L_INF,
    'session': SESSION,
})
builder.config_init_l_2('bim', {})
builder.config_l_2('bim', {
    'magnitude': MAGNITUDE_L_2,
    'alpha': ALPHA_L_2,
    'session': SESSION,
})


builder.config_init_l_inf('cw', {
    'confidence': 1e-6,
    'learning_rate': 1e-2,
})
builder.config_l_inf('cw', {
    'cs': 1e-3,
    'search_steps': 4,
    'binsearch_steps': 10,
})
builder.config_init_l_2('cw', {
    'confidence': 1e-6,
    'learning_rate': 1e-2,
})
builder.config_l_2('cw', {
    'cs': 1.0,
    'search_steps': 2,
    'binsearch_steps': 10,
})


builder.config_init_l_inf('mim', {
    'decay_factor': 1.0
})
builder.config_l_inf('mim', {
    'magnitude': MAGNITUDE_L_INF,
    'alpha': ALPHA_L_INF,
    'session': SESSION,
})
builder.config_init_l_2('mim', {
    'decay_factor': 1.0
})
builder.config_l_2('mim', {
    'magnitude': MAGNITUDE_L_2,
    'alpha': ALPHA_L_2,
    'session': SESSION,
})


builder.iteration(ITERATION)
builder.batch_size(BATCH_SIZE)
builder.no_batch_pred(True)

benchmark = builder.build(SESSION, MODEL,
                          args.method, args.goal, args.distance_metric)

os.makedirs(args.output_dir, exist_ok=True)

for count, (filenames, xs, ys, ys_target) in enumerate(
    load_batches_imagenet_test(
        batch_size=BATCH_SIZE, x_min=MODEL.x_min, x_max=MODEL.x_max,
        x_shape=MODEL.x_shape, x_dtype=MODEL.x_dtype, y_dtype=MODEL.y_dtype,
        start=0, end=1000, label_offset=LABEL_OFFSET,
        return_target_class=True)):
    print(count * BATCH_SIZE, (count + 1) * BATCH_SIZE)
    output_filename = os.path.join(args.output_dir, '%d_rs.npy' % count)
    rs = benchmark.run(xs, ys, ys_target)
    np.save(output_filename, rs)

SESSION.close()
