from realsafe.benchmark.utils import iteartion_benchmark_parser, new_session
from realsafe.benchmark.iteration_benchmark import IterationBenchmarkBuilder
from realsafe.defense.jpeg import jpeg_classifier_differentiable
from realsafe.cifar10.ResNet56 import ResNet56
import tensorflow as tf
import numpy as np

SESSION = tf.Session()


args = iteartion_benchmark_parser()

JPEG_SESSION = new_session()
paras = (0.0, 1.0, (32, 32, 3), tf.float32, tf.int32, 10, 75, JPEG_SESSION)
ResNet56_Jpeg = jpeg_classifier_differentiable(*paras)(ResNet56)
MODEL = ResNet56_Jpeg()
ITERATION = 100
MAGNITUDE_L_INF = 8.0 / 255.0
ALPHA_L_INF = 1.0 / 255.0
SPSA_LR_L_INF = ALPHA_L_INF
MAGNITUDE_L_2 = 1.0
ALPHA_L_2 = 0.15
SPSA_LR_L_2 = 0.005

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
builder.batch_size(100)

benchmark = builder.build(SESSION, MODEL,
                          args.method, args.goal, args.distance_metric)

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
    output = arg_output

    np.save(output, benchmark.run(xs, ys, ys_target))

SESSION.close()
