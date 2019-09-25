from realsafe.cifar10.ResNet56 import ResNet56
from realsafe.benchmark.distortion_benchmark import DistortionBenchmarkBuilder
from realsafe.benchmark.utils import distortion_benchmark_parser
import tensorflow as tf
import numpy as np

SESSION = tf.Session()


args = distortion_benchmark_parser()

MODEL = ResNet56()

MODEL.load(session=SESSION)
ITERATION = 20

builder = DistortionBenchmarkBuilder()
builder.init_distortion_l_2(32.0 * (MODEL.x_max - MODEL.x_min))
builder.init_distortion_l_inf(1.0 * (MODEL.x_max - MODEL.x_min))
builder.search_steps(0)
builder.binsearch_steps(14)
builder.batch_size(100)

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

xs_ph = tf.placeholder(tf.float32, shape=(None, *MODEL.x_shape))
_, labels = MODEL.logits_and_labels(xs_ph)

for arg_xs, arg_ys, arg_ys_target, arg_output in \
        zip(args.xs, args.ys, args.ys_target, args.output):
    print(arg_xs)
    print(arg_ys)
    print(arg_ys_target)
    print(arg_output)

    xs = np.load(arg_xs).astype(MODEL.x_dtype.as_numpy_dtype)
    xs = (xs / 255.0) * (MODEL.x_max - MODEL.x_min) + MODEL.x_min
    ys = np.load(arg_ys).astype(MODEL.y_dtype.as_numpy_dtype)
    ys_target = np.load(arg_ys_target).astype(MODEL.y_dtype.as_numpy_dtype)

    np.save(arg_output, benchmark.run(xs, ys, ys_target))
    ys_adv = SESSION.run(labels, feed_dict={xs_ph: np.load(arg_output)})
    print((ys_adv == ys).astype(np.float32).mean())
    print((ys_adv == ys_target).astype(np.float32).mean())
