import numpy as np
import tensorflow as tf
import random

from realsafe.dataset.cifar10 import load_batches_cifar_test

BATCH_SIZE = 100


def gen_ys_target(ys):
    ts = []

    for y in ys:
        while True:
            t = random.randint(0, 9)
            if y != t:
                ts.append(t)
                break

    return np.array(ts, dtype=np.int32)


for n, (_, xs, ys) in enumerate(load_batches_cifar_test(
    batch_size=BATCH_SIZE, x_min=0.0, x_max=1.0,
    x_dtype=tf.float32, x_shape=(32, 32, 3),
    y_dtype=tf.int32
)):
    xs = xs * 255
    xs = xs.astype(np.uint8)
    ys_target = gen_ys_target(ys)

    np.save("{:05d}_{:05d}_xs".format(
        BATCH_SIZE * n, BATCH_SIZE * (n + 1)), xs)
    np.save("{:05d}_{:05d}_ys".format(
        BATCH_SIZE * n, BATCH_SIZE * (n + 1)), ys)
    np.save("{:05d}_{:05d}_ys_target".format(
        BATCH_SIZE * n, BATCH_SIZE * (n + 1)), ys_target)
