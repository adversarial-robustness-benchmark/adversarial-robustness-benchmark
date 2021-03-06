from __future__ import absolute_import
import numpy as np
import tensorflow as tf
import cv2

from realsafe.attack.base import Attack
from realsafe.model import ClassifierWithLogits
from realsafe.attack.utils import clip_eta


class NAttack(Attack):
    """
    l_2, l_inf
    constrained
    """
    def __init__(self, model, batch_size):
        assert isinstance(model, ClassifierWithLogits)
        if batch_size != 1:
            raise NotImplementedError

        Attack.__init__(self, model, batch_size)

        self.xs_ph = tf.placeholder(model.x_dtype, (None,) + self.model.x_shape)
        self.ys_ph = tf.placeholder(model.y_dtype, (None,))

        logits, self.labels_pred = self.model.logits_and_labels(self.xs_ph)

        # NAttack uses margin logit loss proposed in C&W.
        logit_mask = tf.one_hot(self.ys_ph, self.model.n_class)
        label_logits = tf.reduce_sum(logit_mask * logits, axis=-1)
        highest_nonlabel_logits = tf.reduce_max(logits - logit_mask * 99999,
                                                axis=-1)
        self.loss = highest_nonlabel_logits - label_logits

    def _is_adversarial(self, xs, ys, ys_target, session, goal):
        label = session.run(self.labels_pred, feed_dict={self.xs_ph: xs})
        if goal == "ut" or goal == "tm":
            return label != ys
        else:
            return label == ys_target

    def config(self, **kwargs):
        self.goal = kwargs["goal"]
        self.magnitude = kwargs["magnitude"]
        self.distance_metric = kwargs["distance_metric"]
        self.max_queries = kwargs["max_queries"]
        self.samples_per_draw = kwargs["samples_per_draw"]
        self.sigma = kwargs["sigma"]
        self.lr = kwargs["learning_rate"]
        self.logging = kwargs.get("logging", True)
        self.return_details = kwargs.get("return_details", True)

    def batch_attack(self, xs, ys, ys_target, session):
        assert xs.shape[0] == self.batch_size  # Only run one example
        if ys is not None:
            assert ys.shape[0] == self.batch_size
        if ys_target is not None:
            assert ys_target.shape[0] == self.batch_size
        label = ys if self.goal == "ut" else ys_target

        q = 0

        if self._is_adversarial(xs, ys, ys_target, session, self.goal):
            if self.logging:
                print("Original image is adversarial")
            if self.return_details:
                return xs, q
            else:
                return xs

        def scale(vec, dst_lo, dst_hi, src_lo, src_hi):
            k = (dst_hi - dst_lo) / (src_hi - src_lo)
            b = dst_lo - k * src_lo
            return k * vec + b

        def scale_to_model(vec):
            return scale(vec, self.model.x_min, self.model.x_max, -1.0, 1.0)

        def scale_to_tanh(vec):
            return scale(vec, 1e-6 - 1, 1 - 1e-6,
                         self.model.x_min, self.model.x_max)

        shape = self.model.x_shape
        x_dtype = self.model.x_dtype.as_numpy_dtype
        do_dimension_reduction = (len(shape) == 3)
        if do_dimension_reduction:
            pert_shape = (32, 32, 3)
        else:
            pert_shape = shape
        
        modify = np.random.randn(1, *pert_shape) * 0.001

        while q < self.max_queries:
            pert = np.random.randn(self.samples_per_draw, *pert_shape).astype(x_dtype)
            modify_try = modify + self.sigma * pert
            modify_try = np.array([cv2.resize(modify_try[i], shape[:2]) 
                for i in range(self.samples_per_draw)])

            arctanh_xs = np.arctanh(scale_to_tanh(xs))
            eval_points = scale_to_model(np.tanh(arctanh_xs + modify_try))
            eta = eval_points - xs
            for i in range(self.samples_per_draw):
                eval_points[i] = xs + clip_eta(eta[i], self.distance_metric, self.magnitude)

            losses = session.run(self.loss, feed_dict={
                self.xs_ph: eval_points,
                self.ys_ph: np.repeat(label, self.samples_per_draw)
            })
            normalized_losses = (losses - np.mean(losses)) / (np.std(losses) + 1e-7)
            q += self.samples_per_draw

            grad = normalized_losses.reshape(
                [-1] + [1] * (len(self.model.x_shape))) * pert
            grad = np.mean(grad, axis=0) / self.sigma

            if self.goal != "ut":
                grad = -grad

            modify = modify + self.lr * grad

            modify_test = np.array([cv2.resize(modify[i], shape[:2]) for i in range(modify.shape[0])])
            xs_adv = scale_to_model(np.tanh(arctanh_xs + modify_test))
            xs_adv = xs + clip_eta(xs_adv - xs, self.distance_metric, self.magnitude)
            xs_adv = np.clip(xs_adv, self.model.x_min, self.model.x_max)

            if self.logging:
                xs_adv_label = session.run(self.labels_pred,
                                           feed_dict={self.xs_ph: xs_adv})
                print("queries:{}, loss:{}, learning rate:{}, "
                      "prediction:{}, distortion:{} {}".format(
                    q, np.mean(losses), self.lr, xs_adv_label,
                    np.max(np.abs(xs_adv - xs)), np.linalg.norm(xs_adv - xs)
                ))

            if self._is_adversarial(xs_adv, ys, ys_target, session, self.goal):
                if self.return_details:
                    return xs_adv, q
                else:
                    return xs_adv

        if self.return_details:
            return xs_adv, q
        else:
            return xs_adv