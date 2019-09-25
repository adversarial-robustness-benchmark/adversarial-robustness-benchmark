import tensorflow as tf
import numpy as np
from realsafe.model import ClassifierDifferentiable
from realsafe.utils import tensorflow_op_wrapper


def jpeg_compress(xs, x_min, x_max, session, quality=95):
    # batch_size x width x height x channel
    imgs = (xs * (255.0 / (x_max - x_min))).astype(np.uint8)
    batch_size, width, height, channel = imgs.shape
    rs = np.zeros_like(xs)
    img_ph = tf.placeholder(shape=(width, height, channel), dtype=tf.uint8)
    img_jpeg = tf.image.encode_jpeg(img_ph, format="", quality=quality)
    img_output = tf.image.decode_jpeg(img_jpeg)
    for i in range(len(xs)):
        rs[i] = session.run(img_output, feed_dict={img_ph: imgs[0]}) * (
            (x_max - x_min) / 255.0)
    return rs


def jpeg_classifier_differentiable(x_min, x_max, x_shape, x_dtype, y_dtype,
                                   n_class, jpeg_quality, jpeg_session):
    shape = (None, *x_shape)
    xs_ph = tf.placeholder(tf.float32, shape=shape)
    rs_uint8 = tf.map_fn(
        lambda x: tf.image.decode_jpeg(
            tf.image.encode_jpeg(x, quality=jpeg_quality)),
        tf.cast(xs_ph, tf.uint8))
    rs = tf.cast(rs_uint8, x_dtype)

    x_dtype_np = x_dtype.as_numpy_dtype

    def _jpeg(xs_np):
        return jpeg_session.run(rs, feed_dict={xs_ph: xs_np}).astype(x_dtype_np)

    def _jpeg_grad(xs_np, grad_np):
        return grad_np.astype(x_dtype_np)

    def deco(cls):
        _jpeg_op = tensorflow_op_wrapper(_jpeg, x_dtype, _jpeg_grad, tf.float32,
                                         cls.__name__)

        class Wrapper(ClassifierDifferentiable):
            def __init__(self, *args, **kwargs):
                super(ClassifierDifferentiable, self).__init__(
                    x_min=x_min, x_max=x_max, x_shape=x_shape,
                    x_dtype=x_dtype, y_dtype=y_dtype, n_class=n_class)
                self._model = cls(*args, **kwargs)

            def logits_and_labels(self, xs_ph):
                xs_img = ((xs_ph - x_min) / (x_max - x_min)) * 255.0
                xs_jpeg = _jpeg_op(xs_img)
                xs_jpeg.set_shape(xs_ph.shape)
                xs_model = (xs_jpeg / 255.0) * (x_max - x_min) + x_min
                return self._model.logits_and_labels(xs_model)

            def labels(self, xs_ph):
                _, labels = self.logits_and_labels(xs_ph)
                return labels

            def load(self, **kwargs):
                return self._model.load(**kwargs)

        return Wrapper

    return deco
