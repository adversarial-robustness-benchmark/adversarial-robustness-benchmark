from __future__ import absolute_import

import realsafe.utils
from realsafe.model import ClassifierDifferentiable

import tensorflow as tf
import torch
import keras
from keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense)
from torch.autograd import Variable


def pytorch_classifier_differentiable(x_min, x_max, x_shape, x_dtype,
                                      y_dtype, n_class):
    """
    A decorator for wrapping pytorch model.
    Usage:
        @pytorch_classifier_differentiable(x_min, x_max, x_shape, n_class)
        class PyTorchModel(torch.nn.Module):
            ...
    Example: realsafe.model.mnist.MNISTTorch
    """

    def deco(cls):
        class Wrapper(ClassifierDifferentiable):
            def __init__(self, *args, **kwargs):
                super(ClassifierDifferentiable, self).__init__(
                    x_min=x_min, x_max=x_max, x_shape=x_shape,
                    x_dtype=x_dtype, y_dtype=y_dtype, n_class=n_class)
                self._model = cls(*args, **kwargs)

                def func_logits(xs_np):
                    xs_torch = Variable(torch.from_numpy(
                        xs_np), requires_grad=False)
                    assert isinstance(x_dtype, tf.DType)
                    return self._model(xs_torch).detach().numpy().astype(
                        x_dtype.as_numpy_dtype)

                def func_logits_grad(xs_np, grad_np):
                    xs_torch = Variable(torch.from_numpy(xs_np),
                                        requires_grad=True)
                    logits_torch = self._model(xs_torch)
                    logits_torch.backward(torch.from_numpy(grad_np))
                    return xs_torch.grad.data.numpy().astype(
                        x_dtype.as_numpy_dtype)

                self._logits_op = realsafe.utils.tensorflow_op_wrapper(
                    func_logits, x_dtype, func_logits_grad,
                    tf.float32, cls.__name__)

            def logits_and_labels(self, xs_ph):
                logits = self._logits_op(xs_ph)
                logits.set_shape((None, self.n_class,))
                labels = tf.argmax(logits, 1, output_type=self.y_dtype)
                labels.set_shape((None,))
                return logits, labels

            def labels(self, xs_ph):
                _, labels = self.logits_and_labels(xs_ph)
                return labels

            def load(self, **kwargs):
                return self._model.load(**kwargs)

        return Wrapper

    return deco
