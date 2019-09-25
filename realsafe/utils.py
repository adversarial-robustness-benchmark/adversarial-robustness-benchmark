from __future__ import absolute_import

import tensorflow as tf
import progressbar
import math
import os


def tensorflow_op_wrapper(func, func_dtype, func_grad, func_grad_dtype, name):
    """
    Wrap a python function as a tensorflow operator.
    :param func: The python function to wrap, which accept a numpy array and 
        returns a numpy array.
    :param func_dtype: `func`'s return value's data type.
    :param func_grad:
    :param func_grad_dtype:
    :param name: An unique string to identify the function.
    :return: tensorflow operator
    """

    def grad_wrapper(op, grad):
        return tf.py_func(func_grad, [op.inputs[0], grad], func_grad_dtype)

    grad_name = "PyFuncGrad" + name
    tf.RegisterGradient(grad_name)(grad_wrapper)

    def tensorflow_op(xs):
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": grad_name}):
            return tf.py_func(func, [xs], func_dtype)

    return tensorflow_op


def get_model_path(model_name):
    return "{}/{}".format(
        os.getenv("REALSAFE_MODELS_PATH", "../models"), model_name)


def get_dataset_path(dataset_name):
    dataset_path = "{}/{}".format(
        os.getenv("REALSAFE_MODELS_PATH", "../datasets"), dataset_name)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    return dataset_path


def model_path_exists(model_path):
    import os
    return os.path.exists(model_path)


pbar = None


def show_progress(block_num, block_size, total_size):
    global pbar

    if pbar is None:
        if total_size > 0:
            prefixes = ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi', 'Yi')
            power = min(int(math.log(total_size, 2) / 10), len(prefixes) - 1)
            scaled = float(total_size) / (2 ** (10 * power))
            total_size_str = '{:.1f} {}B'.format(scaled, prefixes[power])
            marker = '*'
            widgets = [
                progressbar.Percentage(),
                ' ', progressbar.DataSize(),
                ' / ', total_size_str,
                ' ', progressbar.Bar(marker=marker),
                ' ', progressbar.ETA(),
                ' ', progressbar.AdaptiveTransferSpeed(),
            ]
            pbar = progressbar.ProgressBar(widgets=widgets,
                                           max_value=total_size)
        else:
            widgets = [
                progressbar.DataSize(),
                ' ', progressbar.Bar(marker=progressbar.RotatingMarker()),
                ' ', progressbar.Timer(),
                ' ', progressbar.AdaptiveTransferSpeed(),
            ]
            pbar = progressbar.ProgressBar(widgets=widgets,
                                           max_value=progressbar.UnknownLength)

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None
