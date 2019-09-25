import tensorflow as tf
import numpy as np

def Randomization(input_tensor, PAD_VALUE=0.0):
    rnd = tf.random_uniform((), 299, 331, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], 
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = 331 - rnd
    w_rem = 331 - rnd
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=PAD_VALUE)
    padded.set_shape((input_tensor.shape[0], 331, 331, 3))
    return padded
    
