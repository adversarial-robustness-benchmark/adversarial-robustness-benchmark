import tensorflow as tf
import numpy as np


def clip_eta(eta, ord, magnitude):
    if ord == 'l_inf':
        eta = np.clip(eta, -magnitude, magnitude)
    elif ord == 'l_2':
        norm = np.maximum(1e-12, np.linalg.norm(eta))
        factor = np.minimum(1, magnitude / norm)
        eta = eta * factor
    else:
        raise NotImplementedError
    return eta


def mean_square_distance(x1, x2, min_, max_):
    return np.mean((x1 - x2) ** 2) / ((max_ - min_) ** 2)


def get_xs_ph(model, batch_size):
    return tf.placeholder(model.x_dtype, (batch_size, *model.x_shape))


def get_ys_ph(model, batch_size):
    return tf.placeholder(model.y_dtype, (batch_size,))
