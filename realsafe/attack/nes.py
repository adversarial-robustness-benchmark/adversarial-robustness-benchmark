from __future__ import absolute_import
import numpy as np
import tensorflow as tf

from realsafe.attack.base import Attack
from realsafe.model import ClassifierWithLogits
from realsafe.attack.utils import clip_eta


class NES(Attack):
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
        
        # NES uses margin logit loss proposed in C&W.
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
        self.min_lr = kwargs["min_lr"]
        self.lr_tuning = kwargs["lr_tuning"]
        self.plateau_length = kwargs["plateau_length"]
        self.logging = kwargs.get("logging", True)
        self.return_details = kwargs.get("return_details", True)

    def batch_attack(self, xs, ys, ys_target, session):
        assert xs.shape[0] == self.batch_size  # Only run one example
        if ys is not None:
            assert ys.shape[0] == self.batch_size
        if ys_target is not None:
            assert ys_target.shape[0] == self.batch_size
        label = ys if self.goal == "ut" else ys_target

        last_loss = []
        q = 0
        lr = self.lr

        if self._is_adversarial(xs, ys, ys_target, session, self.goal):
            if self.logging:
                print("Original image is adversarial")
            if self.return_details:
                return xs, q
            else:
                return xs

        xs_adv = xs.copy()
        while q < self.max_queries:
            pert = np.random.normal(
                size=(self.samples_per_draw // 2,) + self.model.x_shape
                ).astype(self.model.x_dtype.as_numpy_dtype)
            pert = np.concatenate([pert, -pert], axis=0)
            eval_points = xs_adv + self.sigma * pert

            losses = session.run(self.loss, feed_dict={
                self.xs_ph: eval_points,
                self.ys_ph: np.repeat(label, self.samples_per_draw)
            })
            q += self.samples_per_draw

            grad = losses.reshape(
                [-1] + [1] * (len(self.model.x_shape))) * pert
            grad = np.mean(grad, axis=0) / self.sigma

            if self.lr_tuning:
                last_loss.append(np.mean(losses))
                last_loss = last_loss[-self.plateau_length:]
                if len(last_loss) == self.plateau_length:
                    if self.goal == "ut" and last_loss[-1] < last_loss[0]:
                        lr = max(lr / 2, self.min_lr)
                        last_loss = []
                    elif self.goal != "ut" and last_loss[-1] > last_loss[0]:
                        lr = max(lr / 2, self.min_lr)
                        last_loss = []

            if self.goal != "ut":
                grad = -grad

            if self.distance_metric == 'l_inf':
                xs_adv = xs_adv + lr * np.sign(grad)
            elif self.distance_metric == 'l_2':
                grad_norm = np.maximum(1e-12, np.linalg.norm(grad))
                xs_adv = xs_adv + lr * grad / grad_norm
            else:
                raise NotImplementedError

            xs_adv = xs + clip_eta(xs_adv - xs, self.distance_metric,
                                   self.magnitude)
            xs_adv = np.clip(xs_adv, self.model.x_min, self.model.x_max)

            if self.logging:
                xs_adv_label = session.run(self.labels_pred,
                                           feed_dict={self.xs_ph: xs_adv})
                print("queries:{}, loss:{}, learning rate:{},"
                      " prediction:{}, distortion:{} {}".format(
                          q, np.mean(losses), lr, xs_adv_label,
                          np.max(np.abs(xs_adv - xs)
                                 ), np.linalg.norm(xs_adv - xs),
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
