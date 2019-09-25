from __future__ import absolute_import
import numpy as np
import tensorflow as tf
import collections
import cv2

from realsafe.attack.base import Attack
from realsafe.model import Classifier
from realsafe.attack.utils import mean_square_distance as distance


class Evolutionary(Attack):
    """
    l_2
    optimized
    """
    def __init__(self, model, batch_size):
        assert isinstance(model, Classifier)
        if batch_size != 1:
            raise NotImplementedError

        Attack.__init__(self, model=model, batch_size=batch_size)

        self.xs_ph = tf.placeholder(model.x_dtype, (None,) + self.model.x_shape)
        self.labels_pred = self.model.labels(self.xs_ph)

    def _is_adversarial(self, xs, ys, ys_target, session, goal):
        label = session.run(self.labels_pred,
                            feed_dict={self.xs_ph: xs[np.newaxis]})[0]
        if goal == "ut" or goal == "tm":
            return label != ys
        else:
            return label == ys_target

    def log_step(self, step, prediction, distance, sigma, mu, message=''):
        print("Step {}: {:.5e}, prediction = {}, "
              "stepsizes = {:.1e}/{:.1e}: {}".format(
            step, distance, prediction, sigma, mu, message))

    def config(self, **kwargs):
        self.max_queries = kwargs["max_queries"]
        self.goal = kwargs["goal"]

        self.mu = kwargs["mu"]
        self.sigma = kwargs["sigma"]
        self.sample_size = kwargs['sample_size']
        self.logging = kwargs.get("logging", True)
        self.return_details = kwargs.get("return_details", True)

    def batch_attack(self, xs, ys, ys_target, session, starting_point):
        assert xs.shape[0] == self.batch_size  # Only run one example
        xs = np.squeeze(xs, axis=0)
        if ys is not None:
            assert ys.shape[0] == self.batch_size
            ys = np.squeeze(ys, axis=0)
        if ys_target is not None:
            assert ys_target.shape[0] == self.batch_size
            ys_target = np.squeeze(ys_target, axis=0)

        if self._is_adversarial(xs, ys, ys_target, session, self.goal):
            if self.logging:
                print("Original image is adversarial")
            if self.return_details:
                return xs[np.newaxis], np.zeros([self.max_queries + 1])
            else:
                return xs[np.newaxis]
        assert self._is_adversarial(starting_point, ys, ys_target, session, self.goal)

        mu = self.mu
        min_ = self.model.x_min
        max_ = self.model.x_max
        shape = self.model.x_shape
        x_dtype = self.model.x_dtype.as_numpy_dtype
        do_dimension_reduction = (len(shape) == 3)

        xs_adv = starting_point.astype(x_dtype).copy()
        dis = distance(xs_adv, xs, min_, max_)
        stats_adversarial = collections.deque(maxlen=30)
        dis_per_query = np.zeros([self.max_queries + 1])

        if do_dimension_reduction:
            pert_shape = (self.sample_size, self.sample_size, shape[2])
        else:
            pert_shape = shape

        N = 1
        for i in range(len(pert_shape)):
            N *= pert_shape[i]
        K = int(N / 20)

        evolution_path = np.zeros(pert_shape, dtype=x_dtype)
        decay_factor = 0.99
        diagonal_covariance = np.ones(pert_shape, dtype=x_dtype)
        c = 0.001

        xs_adv_label = session.run(
            self.labels_pred, feed_dict={self.xs_ph: xs_adv[np.newaxis]})[0]

        if self.logging:
            self.log_step(0, xs_adv_label, dis, self.sigma, mu)

        dis_per_query[0] = dis

        for step in range(1, self.max_queries + 1):
            unnormalized_source_direction = xs - xs_adv
            source_norm = np.linalg.norm(unnormalized_source_direction)

            selection_probability = \
                diagonal_covariance.reshape(-1) / np.sum(diagonal_covariance)
            selected_indices = np.random.choice(
                N, K, replace=False, p=selection_probability)

            perturbation = np.random.normal(0, 1, pert_shape).astype(x_dtype)
            factor = np.zeros([N], dtype=x_dtype)
            factor[selected_indices] = 1
            perturbation *= factor.reshape(
                pert_shape) * np.sqrt(diagonal_covariance)

            if do_dimension_reduction:
                perturbation_large = cv2.resize(perturbation, shape[:2])
            else:
                perturbation_large = perturbation

            biased = xs_adv + mu * unnormalized_source_direction
            candidate = biased + self.sigma * source_norm * \
                        perturbation_large / np.linalg.norm(perturbation_large)
            candidate = xs - (xs - candidate) / np.linalg.norm(xs - candidate) \
                        * np.linalg.norm(xs - biased)
            candidate = np.clip(candidate, min_, max_)

            is_adversarial = self._is_adversarial(candidate, ys, ys_target,
                                                  session, self.goal)
            stats_adversarial.appendleft(is_adversarial)

            if is_adversarial:
                new_xs_adv = candidate
                new_dis = distance(new_xs_adv, xs, min_, max_)
                evolution_path = decay_factor * evolution_path + \
                                 np.sqrt(1 - decay_factor ** 2) * perturbation
                diagonal_covariance = (1 - c) * diagonal_covariance + \
                                      c * (evolution_path ** 2)
            else:
                new_xs_adv = None

            message = ''
            if new_xs_adv is not None:
                abs_improvement = dis - new_dis
                rel_improvement = abs_improvement / dis
                message = 'd. reduced by {:.2f}% ({:.4e})'.format(
                    rel_improvement * 100, abs_improvement)

                xs_adv = new_xs_adv
                dis = new_dis

            dis_per_query[step] = dis
            xs_adv_label = session.run(
                self.labels_pred, feed_dict={self.xs_ph: xs_adv[np.newaxis]})[0]

            if self.logging:
                self.log_step(step, xs_adv_label, dis, self.sigma, mu, message)

            if len(stats_adversarial) == stats_adversarial.maxlen:
                p_step = np.mean(stats_adversarial)
                n_step = len(stats_adversarial)
                mu *= np.exp(p_step - 0.2)
                stats_adversarial.clear()

        if self.return_details:
            return xs_adv[np.newaxis], dis_per_query
        else:
            return xs_adv[np.newaxis]
