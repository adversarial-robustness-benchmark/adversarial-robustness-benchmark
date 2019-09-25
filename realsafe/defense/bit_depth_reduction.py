import tensorflow as tf
import numpy as np
from realsafe.model import ClassifierDifferentiable
from realsafe.utils import tensorflow_op_wrapper

def bit_depth_reduction(xs, x_min, x_max, step_num, alpha):
    steps = x_min + np.arange(1, step_num, dtype=np.float32) / step_num * (x_max - x_min)
    steps = steps.reshape([1,1,1,1,step_num-1])
    tf_steps = tf.constant(steps, dtype=tf.float32)

    inputs = tf.expand_dims(xs, 4)
    quantized_inputs = x_min + tf.reduce_sum(tf.sigmoid(alpha * (inputs - tf_steps)), axis=4) / (step_num-1) * (x_max - x_min)
    return quantized_inputs


def bit_depth_reduction_classifier_differentiable(x_min, x_max, x_shape, x_dtype, y_dtype,
                                                  n_class, step_num, bit_session):
    """
    A decorator for wrapping a differentiable classifier with jpeg defense.
    Usage:
        paras = (0.0, 1.0, (28, 28, 1), tf.float32, tf.int32, 10, session)
        MNISTFcJpeg = jpeg_classifier_differentiable(*paras)(MNISTFc)
    """
    shape = (None, *x_shape)
    xs_ph = tf.placeholder(tf.float32, shape=shape)
    rs = bit_depth_reduction(xs_ph, x_min, x_max, step_num, alpha=1e6)

    x_dtype_np = x_dtype.as_numpy_dtype

    def _bit(xs_np):
        return bit_session.run(rs, feed_dict={xs_ph: xs_np}).astype(x_dtype_np)

    def _bit_grad(xs_np, grad_np):
        return grad_np.astype(x_dtype_np)

    def deco(cls):
        _bit_op = tensorflow_op_wrapper(_bit, x_dtype, _bit_grad, tf.float32,
                                        cls.__name__)

        class Wrapper(ClassifierDifferentiable):
            def __init__(self, *args, **kwargs):
                super(ClassifierDifferentiable, self).__init__(
                    x_min=x_min, x_max=x_max, x_shape=x_shape,
                    x_dtype=x_dtype, y_dtype=y_dtype, n_class=n_class)
                self._model = cls(*args, **kwargs)

            def xs_bit(self, xs_ph):
                xs_bit = _bit_op(xs_ph)
                xs_bit.set_shape(xs_ph.shape)
                return xs_bit

            def logits_and_labels(self, xs_ph):
                xs_bit = _bit_op(xs_ph)
                xs_bit.set_shape(xs_ph.shape)
                return self._model.logits_and_labels(xs_bit)

            def labels(self, xs_ph):
                _, labels = self.logits_and_labels(xs_ph)
                return labels

            def load(self, **kwargs):
                return self._model.load(**kwargs)

        return Wrapper

    return deco
