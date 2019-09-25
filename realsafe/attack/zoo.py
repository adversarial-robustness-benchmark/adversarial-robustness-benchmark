from __future__ import absolute_import, division

from realsafe.attack import Attack
from realsafe.model import ClassifierWithLogits
import sys
import tensorflow as tf
import numpy as np
import scipy.misc
import math
import time


def coordinate_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr,
                    real_modifier, up, down, lr, adam_epoch, beta1, beta2,
                    proj):
    for i in range(batch_size):
        grad[i] = (losses[i * 2 + 1] - losses[i * 2 + 2]) / 0.0002

        # ADAM update
    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * grad
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (grad * grad)
    vt_arr[indice] = vt
    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2, epoch))) / (1 - np.power(beta1, epoch))
    m = real_modifier.reshape(-1)
    old_val = m[indice]
    old_val -= lr * corr * mt / (np.sqrt(vt) + 1e-8)
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
    m[indice] = old_val
    adam_epoch[indice] = epoch + 1


def coordinate_Newton(losses, indice, grad, hess, batch_size, mt_arr, vt_arr,
                      real_modifier, up, down, lr, adam_epoch, beta1, beta2,
                      proj):
    cur_loss = losses[0]
    for i in range(batch_size):
        grad[i] = (losses[i * 2 + 1] - losses[i * 2 + 2]) / 0.0002
        hess[i] = (losses[i * 2 + 1] - 2 * cur_loss + losses[i * 2 + 2]) / (
                0.0001 * 0.0001)

    # negative hessian cannot provide second order information, just do a
    # gradient descent
    hess[hess < 0] = 1.0
    hess[hess < 0.1] = 0.1
    m = real_modifier.reshape(-1)
    old_val = m[indice]
    old_val -= lr * grad / hess
    # set it back to [-0.5, +0.5] region
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])

    m[indice] = old_val


def coordinate_Newton_ADAM(losses, indice, grad, hess, batch_size, mt_arr,
                           vt_arr, real_modifier, up, down, lr, adam_epoch,
                           beta1, beta2, proj):
    cur_loss = losses[0]
    for i in range(batch_size):
        grad[i] = (losses[i * 2 + 1] - losses[i * 2 + 2]) / 0.0002
        hess[i] = (losses[i * 2 + 1] - 2 * cur_loss + losses[i * 2 + 2]) / (
                0.0001 * 0.0001)

    hess_indice = (hess >= 0)
    adam_indice = (hess < 0)
    hess[hess < 0] = 1.0
    hess[hess < 0.1] = 0.1
    # Newton's Method
    m = real_modifier.reshape(-1)
    old_val = m[indice[hess_indice]]
    old_val -= lr * grad[hess_indice] / hess[hess_indice]
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice[hess_indice]]),
                             down[indice[hess_indice]])
    m[indice[hess_indice]] = old_val
    # ADMM
    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * grad
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (grad * grad)
    vt_arr[indice] = vt
    # epoch is an array; for each index we can have a different epoch number
    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2, epoch[adam_indice]))) / (
            1 - np.power(beta1, epoch[adam_indice]))
    old_val = m[indice[adam_indice]]
    old_val -= lr * corr * mt[adam_indice] / (np.sqrt(vt[adam_indice]) + 1e-8)
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice[adam_indice]]),
                             down[indice[adam_indice]])
    m[indice[adam_indice]] = old_val
    adam_epoch[indice] = epoch + 1


class ZOO(Attack):
    """
    l_2
    optimized
    """

    def __init__(self, sess, model, batch_size=1, confidence=0, targeted=True,
                 learning_rate=2e-3,
                 binary_search_steps=1, max_iterations=10000, print_every=100,
                 early_stop_iters=0,
                 abort_early=True, initial_const=0.5, use_log=False,
                 use_tanh=True, adam_beta1=0.9,
                 adam_beta2=0.999, reset_adam_after_found=False, solver="adam",
                 start_iter=0,
                 init_size=32, use_importance=True):
        assert isinstance(model, ClassifierWithLogits)
        Attack.__init__(self, model, 1)

        _, image_size, num_channels = model.x_shape
        self.num_labels = model.n_class

        self.model = model
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.print_every = print_every
        self.early_stop_iters = early_stop_iters if early_stop_iters != 0 else \
            max_iterations // 10

        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.start_iter = start_iter
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.resize_init_size = init_size
        self.use_importance = use_importance

        self.small_x = image_size
        self.small_y = image_size
        self.use_tanh = use_tanh

        self.repeat = binary_search_steps >= 10

        shape = (None, image_size, image_size, num_channels)
        single_shape = (image_size, image_size, num_channels)
        small_single_shape = (self.small_x, self.small_y, num_channels)

        self.modifier = tf.placeholder(tf.float32, shape=(
            None, image_size, image_size, num_channels))
        # no resize
        self.scaled_modifier = self.modifier
        # the real variable, initialized to 0
        self.real_modifier = np.zeros((1,) + small_single_shape,
                                      dtype=np.float32)

        self.timg = tf.Variable(np.zeros(single_shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros(self.num_labels), dtype=tf.float32)
        self.const = tf.Variable(0.0, dtype=tf.float32)

        self.assign_timg = tf.placeholder(tf.float32, single_shape)
        self.assign_tlab = tf.placeholder(tf.float32, self.num_labels)
        self.assign_const = tf.placeholder(tf.float32)

        if use_tanh:
            self.newimg = tf.tanh(self.scaled_modifier + self.timg) / 2
        else:
            self.newimg = self.scaled_modifier + self.timg

        self.output, _ = self.model.logits_and_labels(self.newimg)

        if use_tanh:
            self.l2dist = tf.reduce_sum(
                tf.square(self.newimg - tf.tanh(self.timg) / 2), [1, 2, 3])
        else:
            self.l2dist = tf.reduce_sum(tf.square(self.newimg - self.timg),
                                        [1, 2, 3])

        self.real = tf.reduce_sum((self.tlab) * self.output, 1)
        self.other = tf.reduce_max(
            (1 - self.tlab) * self.output - (self.tlab * 10000), 1)

        if self.TARGETED:
            if use_log:
                loss1 = tf.maximum(0.0, tf.log(self.other + 1e-30) - tf.log(
                    self.real + 1e-30))
            else:
                loss1 = tf.maximum(0.0,
                                   self.other - self.real + self.CONFIDENCE)
        else:
            if use_log:
                loss1 = tf.maximum(0.0, tf.log(self.real + 1e-30) - tf.log(
                    self.other + 1e-30))
            else:
                loss1 = tf.maximum(0.0,
                                   self.real - self.other + self.CONFIDENCE)

        # sum up the losses (output is a vector of #batch_size)
        self.loss2 = self.l2dist
        self.loss1 = self.const * loss1
        self.loss = self.loss1 + self.loss2

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))

        # prepare the list of all valid variables
        var_size = self.small_x * self.small_y * num_channels
        self.use_var_len = var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype=np.int32)
        self.used_var_list = np.zeros(var_size, dtype=np.int32)
        self.sample_prob = np.ones(var_size, dtype=np.float32) / var_size

        # upper and lower bounds for the modifier
        self.modifier_up = np.zeros(var_size, dtype=np.float32)
        self.modifier_down = np.zeros(var_size, dtype=np.float32)

        # random permutation for coordinate update
        self.perm = np.random.permutation(var_size)
        self.perm_index = 0

        # ADAM status
        self.mt = np.zeros(var_size, dtype=np.float32)
        self.vt = np.zeros(var_size, dtype=np.float32)
        self.beta1 = adam_beta1
        self.beta2 = adam_beta2
        self.reset_adam_after_found = reset_adam_after_found
        self.adam_epoch = np.ones(var_size, dtype=np.int32)
        self.stage = 0
        # variables used during optimization process
        self.grad = np.zeros(batch_size, dtype=np.float32)
        self.hess = np.zeros(batch_size, dtype=np.float32)
        # for testing
        self.grad_op = tf.gradients(self.loss, self.modifier)

        # set solver
        solver = solver.lower()
        self.solver_name = solver
        if solver == "adam":
            self.solver = coordinate_ADAM
        elif solver == "newton":
            self.solver = coordinate_Newton
        elif solver == "adam_newton":
            self.solver = coordinate_Newton_ADAM
        elif solver != "fake_zero":
            print("unknown solver", solver)
            self.solver = coordinate_ADAM
        print("Using", solver, "solver")

    def max_pooling(self, image, size):
        img_pool = np.copy(image)
        img_x = image.shape[0]
        img_y = image.shape[1]
        for i in range(0, img_x, size):
            for j in range(0, img_y, size):
                img_pool[i:i + size, j:j + size] = np.max(
                    image[i:i + size, j:j + size])
        return img_pool

    def get_new_prob(self, prev_modifier, gen_double=False):
        prev_modifier = np.squeeze(prev_modifier)
        old_shape = prev_modifier.shape
        if gen_double:
            new_shape = (old_shape[0] * 2, old_shape[1] * 2, old_shape[2])
        else:
            new_shape = old_shape
        prob = np.empty(shape=new_shape, dtype=np.float32)
        for i in range(prev_modifier.shape[2]):
            image = np.abs(prev_modifier[:, :, i])
            image_pool = self.max_pooling(image, old_shape[0] // 8)
            if gen_double:
                prob[:, :, i] = scipy.misc.imresize(image_pool, 2.0, 'nearest',
                                                    mode='F')
            else:
                prob[:, :, i] = image_pool
        prob /= np.sum(prob)
        return prob

    def fake_blackbox_optimizer(self):
        true_grads, losses, l2s, loss1, loss2, scores, nimgs = self.sess.run(
            [self.grad_op, self.loss, self.l2dist, self.loss1, self.loss2,
             self.output, self.newimg],
            feed_dict={self.modifier: self.real_modifier})
        # ADAM update
        grad = true_grads[0].reshape(-1)

        epoch = self.adam_epoch[0]
        mt = self.beta1 * self.mt + (1 - self.beta1) * grad
        vt = self.beta2 * self.vt + (1 - self.beta2) * np.square(grad)
        corr = (math.sqrt(1 - self.beta2 ** epoch)) / (1 - self.beta1 ** epoch)

        m = self.real_modifier.reshape(-1)
        m -= self.LEARNING_RATE * corr * (mt / (np.sqrt(vt) + 1e-8))
        self.mt = mt
        self.vt = vt

        if not self.use_tanh:
            m_proj = np.maximum(np.minimum(m, self.modifier_up),
                                self.modifier_down)
            np.copyto(m, m_proj)
        self.adam_epoch[0] = epoch + 1
        return losses[0], l2s[0], loss1[0], loss2[0], scores[0], nimgs[0]

    def blackbox_optimizer(self, iteration):
        var = np.repeat(self.real_modifier, self.batch_size * 2 + 1, axis=0)
        var_size = self.real_modifier.size

        if self.use_importance:
            var_indice = np.random.choice(self.var_list.size, self.batch_size,
                                          replace=False, p=self.sample_prob)
        else:
            var_indice = np.random.choice(self.var_list.size, self.batch_size,
                                          replace=False)
        indice = self.var_list[var_indice]

        for i in range(self.batch_size):
            var[i * 2 + 1].reshape(-1)[indice[i]] += 0.0001
            var[i * 2 + 2].reshape(-1)[indice[i]] -= 0.0001
        losses, l2s, loss1, loss2, scores, nimgs = self.sess.run(
            [self.loss, self.l2dist, self.loss1, self.loss2, self.output,
             self.newimg], feed_dict={self.modifier: var})

        self.solver(losses, indice, self.grad, self.hess, self.batch_size,
                    self.mt, self.vt, self.real_modifier, self.modifier_up,
                    self.modifier_down, self.LEARNING_RATE, self.adam_epoch,
                    self.beta1, self.beta2, not self.use_tanh)

        if self.real_modifier.shape[0] > self.resize_init_size:
            self.sample_prob = self.get_new_prob(self.real_modifier)
            self.sample_prob = self.sample_prob.reshape(var_size)

        return losses[0], l2s[0], loss1[0], loss2[0], scores[0], nimgs[0]

    def attack(self, imgs, targets):
        r = []
        for i in range(len(imgs)):
            print('Image', i)
            o_attack, _ = self.attack_batch(imgs[i], targets[i])
            r.append(o_attack)

        return np.array(r)

    # only accepts 1 image at a time.
    def attack_batch(self, img, lab):
        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)

            if self.TARGETED:
                return x == y
            else:
                return x != y

        # remove the extra batch dimension
        if len(img.shape) == 4:
            img = img[0]
        if len(lab.shape) == 2:
            lab = lab[0]

        lab = (np.arange(self.num_labels) == lab).astype(np.float32)
        # convert to tanh-space
        if self.use_tanh:
            img = np.arctanh(img * 1.999999)

        # set the lower and upper bounds accordingly
        lower_bound = 0.0
        CONST = self.initial_const
        upper_bound = 1e10

        # convert img to float32 to avoid numba error
        img = img.astype(np.float32)

        # set the upper and lower bounds for the modifier
        if not self.use_tanh:
            self.modifier_up = 0.5 - img.reshape(-1)
            self.modifier_down = -0.5 - img.reshape(-1)

        # clear the modifier
        self.real_modifier.fill(0.0)

        # the best l2, score, and image attack
        o_best_const = CONST
        o_bestl2 = 1e10
        o_bestscore = -1
        o_bestattack = img

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print(o_bestl2)
            bestl2 = 1e10
            bestscore = -1

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: img,
                                       self.assign_tlab: lab,
                                       self.assign_const: CONST})

            prev = 1e6
            train_timer = 0.0
            last_loss1 = 1.0

            self.real_modifier.fill(0.0)
            # reset ADAM status
            self.mt.fill(0.0)
            self.vt.fill(0.0)
            self.adam_epoch.fill(1)
            self.stage = 0
            multiplier = 1
            eval_costs = 0
            if self.solver_name != "fake_zero":
                multiplier = 24
            for iteration in range(self.start_iter, self.MAX_ITERATIONS):

                if iteration % (self.print_every) == 0:
                    loss, real, other, loss1, loss2 = self.sess.run(
                        (self.loss, self.real, self.other,
                         self.loss1, self.loss2),
                        feed_dict={self.modifier: self.real_modifier})
                    print(
                        "[STATS][L2] iter = {}, cost = {}, time = {:.3f}, "
                        "size = {}, loss = {:.5g}, real = {:.5g}, "
                        "other = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}".format(
                            iteration, eval_costs, train_timer,
                            self.real_modifier.shape, loss[0], real[0],
                            other[0], loss1[0], loss2[0])
                    )
                    sys.stdout.flush()

                attack_begin_time = time.time()
                # perform the attack 
                if self.solver_name == "fake_zero":
                    l, l2, loss1, loss2, score, nimg = \
                        self.fake_blackbox_optimizer()
                else:
                    l, l2, loss1, loss2, score, nimg = \
                        self.blackbox_optimizer(iteration)

                if self.solver_name == "fake_zero":
                    eval_costs += np.prod(self.real_modifier.shape)
                else:
                    eval_costs += self.batch_size

                # reset ADAM states when a valid example has been found
                if loss1 == 0.0 and last_loss1 != 0.0 and self.stage == 0:
                    if self.reset_adam_after_found:
                        self.mt.fill(0.0)
                        self.vt.fill(0.0)
                        self.adam_epoch.fill(1)
                    self.stage = 1
                last_loss1 = loss1

                if self.ABORT_EARLY and iteration % self.early_stop_iters == 0:
                    if l > prev * .9999:
                        print("Early stopping because there is no improvement")
                        break
                    prev = l

                if l2 < bestl2 and compare(score, np.argmax(lab)):
                    bestl2 = l2
                    bestscore = np.argmax(score)
                if l2 < o_bestl2 and compare(score, np.argmax(lab)):
                    if o_bestl2 == 1e10:
                        print(
                            "[STATS][L3](First valid attack found!) iter = {}, "
                            "cost = {}, time = {:.3f}, size = {}, "
                            "loss = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}, "
                            "l2 = {:.5g}".format(
                                iteration, eval_costs, train_timer,
                                self.real_modifier.shape, l, loss1, loss2, l2)
                        )
                        sys.stdout.flush()
                    o_bestl2 = l2
                    o_bestscore = np.argmax(score)
                    o_bestattack = nimg
                    o_best_const = CONST

                train_timer += time.time() - attack_begin_time

            # adjust the constant as needed
            if compare(bestscore, np.argmax(lab)) and bestscore != -1:
                print('old constant: ', CONST)
                upper_bound = min(upper_bound, CONST)
                if upper_bound < 1e9:
                    CONST = (lower_bound + upper_bound) / 2
                print('new constant: ', CONST)
            else:
                print('old constant: ', CONST)
                lower_bound = max(lower_bound, CONST)
                if upper_bound < 1e9:
                    CONST = (lower_bound + upper_bound) / 2
                else:
                    CONST *= 10
                print('new constant: ', CONST)
        return o_bestattack, o_best_const
