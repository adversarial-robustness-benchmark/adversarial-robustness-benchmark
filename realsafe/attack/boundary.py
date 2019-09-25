from __future__ import absolute_import
import numpy as np
import tensorflow as tf
import collections

from realsafe.attack.base import Attack
from realsafe.model import Classifier
from realsafe.attack.utils import mean_square_distance as distance


class Boundary(Attack):
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

    def log_step(self, step, prediction, distance, spherical_step, source_step,
                 message=""):
        print("Step {}: {:.5e}, prediction = {}, "
              "stepsizes = {:.1e}/{:.1e}: {}".format(
            step, distance, prediction, spherical_step, source_step, message))

    def config(self, **kwargs):
        self.iteration = kwargs["iteration"]
        self.max_directions = kwargs["max_directions"]
        self.max_queries = kwargs["max_queries"]

        self.spherical_step = kwargs["spherical_step"]
        self.source_step = kwargs["source_step"]
        self.step_adaptation = kwargs["step_adaptation"]
        self.goal = kwargs["goal"]
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

        min_ = self.model.x_min
        max_ = self.model.x_max
        shape = self.model.x_shape
        x_dtype = self.model.x_dtype.as_numpy_dtype

        spherical_step = self.spherical_step
        source_step = self.source_step
        step_adaptation = self.step_adaptation

        xs_adv = starting_point.astype(x_dtype).copy()
        dis = distance(xs_adv, xs, min_, max_)
        stats_spherical_adversarial = collections.deque(maxlen=100)
        stats_step_adversarial = collections.deque(maxlen=30)
        dis_per_query = np.zeros([self.max_queries + 1])

        xs_adv_label = session.run(
            self.labels_pred, feed_dict={self.xs_ph: xs_adv[np.newaxis]})[0]
        
        if self.logging:
            self.log_step(0, xs_adv_label, dis, spherical_step, source_step)
        
        dis_per_query[0] = dis

        q = 0
        last_q = 0

        for step in range(1, self.iteration):
            unnormalized_source_direction = xs - xs_adv
            source_norm = np.linalg.norm(unnormalized_source_direction)
            source_direction = unnormalized_source_direction / source_norm

            do_spherical = (step % 10 == 0)

            for i in range(self.max_directions):
                perturbation = np.random.normal(0, 1, shape).astype(x_dtype)
                dot = np.vdot(perturbation, source_direction)
                perturbation -= dot * source_direction
                perturbation *= spherical_step * source_norm / np.linalg.norm(
                    perturbation)

                D = 1 / np.sqrt(spherical_step ** 2 + 1)
                direction = perturbation - unnormalized_source_direction
                spherical_candidate = xs + D * direction
                spherical_candidate = np.clip(spherical_candidate, min_, max_)

                new_source_direction = xs - spherical_candidate
                new_source_direction_norm = np.linalg.norm(new_source_direction)
                length = source_step * source_norm

                deviation = new_source_direction_norm - source_norm
                length += deviation
                length = max(0, length)

                length = length / new_source_direction_norm
                candidate = spherical_candidate + length * new_source_direction
                candidate = np.clip(candidate, min_, max_)

                if do_spherical:
                    spherical_is_adversarial = self._is_adversarial(
                        spherical_candidate, ys, ys_target, session, self.goal)
                    q += 1
                    stats_spherical_adversarial.appendleft(
                        spherical_is_adversarial)

                    if not spherical_is_adversarial:
                        continue

                is_adversarial = self._is_adversarial(candidate, ys, ys_target,
                                                      session, self.goal)
                q += 1
                if do_spherical:
                    stats_step_adversarial.appendleft(is_adversarial)

                if not is_adversarial:
                    continue

                new_xs_adv = candidate
                new_dis = distance(new_xs_adv, xs, min_, max_)
                break
            else:
                new_xs_adv = None

            dis_per_query[last_q:min(q, self.max_queries + 1)] = dis

            message = ''
            if new_xs_adv is not None:
                abs_improvement = dis - new_dis
                rel_improvement = abs_improvement / dis
                message = 'd. reduced by {:.2f}% ({:.4e})'.format(
                    rel_improvement * 100, abs_improvement)

                xs_adv = new_xs_adv
                dis = new_dis

            xs_adv_label = session.run(self.labels_pred, feed_dict={
                                         self.xs_ph: xs_adv[np.newaxis]})[0]

            if self.logging:
                self.log_step(step, xs_adv_label, dis, spherical_step, source_step,
                              message)

            if (len(stats_step_adversarial) ==
                stats_step_adversarial.maxlen) and \
                    (len(stats_spherical_adversarial) ==
                     stats_spherical_adversarial.maxlen):

                p_spherical = np.mean(stats_spherical_adversarial)
                p_step = np.mean(stats_step_adversarial)
                n_spherical = len(stats_spherical_adversarial)
                n_step = len(stats_step_adversarial)

                if p_spherical > 0.5:
                    message = 'Boundary too linear, increasing steps:'
                    spherical_step *= step_adaptation
                    source_step *= step_adaptation
                elif p_spherical < 0.2:
                    message = 'Boundary too non-linear, decreasing steps:'
                    spherical_step /= step_adaptation
                    source_step /= step_adaptation
                else:
                    message = None

                if message is not None:
                    stats_spherical_adversarial.clear()
                    if self.logging:
                        print(" {} {:.2f} ({:3d}), {:.2f} ({:3d})".format(
                            message, p_spherical, n_spherical, p_step, n_step))

                if p_step > 0.5:
                    message = 'Success rate too high, increasing source step:'
                    source_step *= step_adaptation
                elif p_step < 0.2:
                    message = 'Success rate too low, decreasing source step:'
                    source_step /= step_adaptation
                else:
                    message = None

                if message is not None:
                    stats_step_adversarial.clear()
                    if self.logging:
                        print(" {} {:.2f} ({:3d}), {:.2f} ({:3d})".format(
                            message, p_spherical, n_spherical, p_step, n_step))

            last_q = q
            if q > self.max_queries:
                break

        if self.return_details:
            return xs_adv[np.newaxis], dis_per_query
        else:
            return xs_adv[np.newaxis]
