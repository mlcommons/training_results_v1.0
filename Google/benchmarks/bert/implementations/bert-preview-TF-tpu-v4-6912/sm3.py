"""SM3 optimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np

import tensorflow.compat.v1 as tf


class SM3Optimizer(tf.train.Optimizer):
  """SM3 variant SM3-II from https://arxiv.org/abs/1901.11150 ."""

  def __init__(self,
               learning_rate,
               momentum=0.0,
               beta2=1.0,
               weight_decay_rate=0.0,
               exclude_from_weight_decay=None,
               name="SM3"):
    """Construct a new SM3 optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      momentum: A `Tensor` or a floating point value. Momentum is not applied to
        sparse updates.
      beta2: 1.0 means sum of gradients squares, while less
        than 1.0 switches to RMSProp style exponential moving averages of the
        second moments.
      weight_decay_rate: A floating point value. Weight decay rate.
      exclude_from_weight_decay: A list of string to be excluded from weight
         decay.
      name: Optional name prefix for the operations created when applying
        gradients.
    """
    super(SM3Optimizer, self).__init__(False, name)
    self._learning_rate = learning_rate
    self._momentum = momentum
    self._beta2 = beta2
    self._weight_decay_rate = weight_decay_rate
    self._exclude_from_weight_decay = exclude_from_weight_decay

  def _create_slots(self, var_list):
    for v in var_list:
      with tf.colocate_with(v):
        if self._momentum > 0:
          self._zeros_slot(v, "momentum", self._name)
        shape = np.array(v.get_shape())
        var_rank = len(shape)
        # We special case vectors and scalars as we can run the diagonal adagrad
        # update for those parameters.
        if var_rank > 1:
          for i, d in enumerate(shape):
            d_tensor = tf.convert_to_tensor(d)
            diag_init = tf.zeros([d_tensor])
            _ = self._get_or_make_slot(v, diag_init, "accumulator_" + str(i),
                                       self._name)
        else:
          _ = self._zeros_slot(v, "accumulator", self._name)

  def _prepare(self):
    learning_rate = self._call_if_callable(self._learning_rate)
    self._learning_rate_tensor = tf.convert_to_tensor(
        learning_rate, name="learning_rate")
    momentum = self._call_if_callable(self._momentum)
    self._momentum_tensor = tf.convert_to_tensor(momentum, name="momentum")

  def _get_expanded_shape(self, shape, i):
    rank = len(shape)
    # Replaces a `shape` of [M, N, K] with 1 in all dimensions except for i.
    # For eg: i = 1 returns [1, N, 1].
    return [1] * i + [shape[i]] + [1] * (rank - i - 1)

  def _compute_updated_accumulators(self, accumulators, rank):
    # Update rule is:
    #  A = min(A1[i], A2[j], A3[j]) + G[i, j, k]^2
    current_accumulator = accumulators[0]
    for i in range(1, rank):
      # Compute the minimum accumulator value which is a tighter bound to
      # diagonal adagrad's accumulator value.
      current_accumulator = tf.minimum(current_accumulator, accumulators[i])
    return current_accumulator

  def _apply_dense(self, grad, var):
    # SM3 upper bounds  the diagonal adagrads accumulators using fewer
    # parameters.
    #
    # Tensor `T` of shape [M, N, K].
    #
    # `G` be its gradient of shape [M, N, K]
    #
    # SM3 keeps around three accumulators A1, A2, A3 of size M, N, K
    # respectively.
    #
    # `A` be the accumulator of shape [M, N, K]. `A` is not materialized until
    #   its needed for every step, and is approximated by A1, A2, A3.
    #
    # At every gradient update step the accumulators satisify:
    #   A1_t[i] >= Sum_{s <= t} G_t[i, j, k]^2 for all j, k.
    #   A2_t[j] >= Sum_{s <= t} G_t[i, j, k]^2 for all i, k.
    #   A3_t[k] >= Sum_{s <= t} G_t[i, j, k]^2 for all i, j.
    # The RHS is the Adagrad accumulator.
    #
    # For every step we materialize the tensor `A` based on accumulated tensors
    # A1, A2 and A3.
    #
    #  A = min(A1[i], A2[j], A3[j]) + G[i, j, k]^2
    #
    # SM3 preconditioned gradient is
    #
    #  preconditioned G = A^{-0.5} * G
    #
    # We then update the individual accumulator factors as:
    #
    #  A1[i] = max_{j, k} A[i, j, k]
    #  A2[j] = max_{i, k} A[i, j, k]
    #  A3[k] = max_{i, j} A[i, j, k]
    #
    shape = np.array(var.get_shape())
    var_rank = len(shape)
    if var_rank > 1:
      accumulator_list = [
          self.get_slot(var, "accumulator_" + str(i)) for i in range(var_rank)
      ]
      reshaped_accumulators = [
          tf.reshape(accumulator_list[i], self._get_expanded_shape(shape, i))
          for i in range(var_rank)
      ]
      current_accumulator = self._compute_updated_accumulators(
          reshaped_accumulators, var_rank)
      if self._beta2 == 1.0:
        current_accumulator += grad * grad
      else:
        current_accumulator = (
            self._beta2 * current_accumulator + (1 - self._beta2) * grad * grad)
    else:
      accumulator = self.get_slot(var, "accumulator")
      if self._beta2 == 1.0:
        current_accumulator = tf.assign_add(accumulator, grad * grad)
      else:
        current_accumulator = tf.assign(
            accumulator,
            self._beta2 * accumulator + (1 - self._beta2) * (grad * grad))
    accumulator_inv_sqrt = tf.rsqrt(current_accumulator + 1e-30)
    scaled_g = (1.0 - self._momentum_tensor) * (grad * accumulator_inv_sqrt)
    all_updates = []

    with tf.control_dependencies([scaled_g]):
      if var_rank > 1:
        # Updates individual accumulator factors as:
        #  A1[i] = max_{j, k} A[i, j, k]
        #  A2[j] = max_{i, k} A[i, j, k]
        #  A3[k] = max_{i, j} A[i, j, k]
        for i, accumulator in enumerate(accumulator_list):
          axes = list(range(i)) + list(range(i + 1, var_rank))
          dim_accumulator = tf.reduce_max(current_accumulator, axis=axes)
          updates = tf.assign(accumulator, dim_accumulator)
          all_updates.append(updates)

    with tf.control_dependencies(all_updates):
      if self._momentum > 0:
        gbar = self.get_slot(var, "momentum")
        update = tf.assign_add(gbar,
                               gbar * (self._momentum_tensor - 1.0) + scaled_g)
      else:
        update = scaled_g
      if self._do_use_weight_decay(var.name):
        update = update + self._weight_decay_rate * var

      return tf.assign_sub(var, self._learning_rate_tensor * update)

  def _apply_sparse_shared(self, grad_values, grad_indices, var):
    # Note that Momentum is not applied for sparse updates. There are couple of
    # issues with using momentum for sparse update. Given a momentum of 0.9,
    # most of the infrequent updates (less than 1/100) will get effective
    # learning rate reduced by 0.1 (coming from momentum) which is the main
    # problem. Beyond that for one's that are have more than 1/100 frequency has
    # really akward momentum semantics (which will span across batches).

    shape = np.array(var.get_shape())
    var_rank = len(shape)
    # For sparse case, we only update the accumulator representing the sparse
    # dimension. In this case SM3 is similar to isotropic adagrad but with
    # better bound (due to the max operator).
    if var_rank > 1:
      accumulator = self.get_slot(var, "accumulator_" + str(0))
      current_accumulator = tf.gather(accumulator, grad_indices)
      expanded_shape = tf.concat([[tf.shape(current_accumulator)[0]], [1] *
                                  (var_rank - 1)], 0)
      current_accumulator = tf.reshape(current_accumulator, expanded_shape)
      current_accumulator += grad_values * grad_values
    else:
      accumulator = self.get_slot(var, "accumulator")
      current_accumulator = tf.scatter_add(accumulator, grad_indices,
                                           grad_values * grad_values)

    accumulator_inv_sqrt = tf.rsqrt(current_accumulator + 1e-30)
    scaled_g = (grad_values * accumulator_inv_sqrt)
    updates = []
    with tf.control_dependencies([scaled_g]):
      if var_rank > 1:
        axes = list(range(1, var_rank))
        dim_accumulator = tf.reduce_max(current_accumulator, axis=axes)
        updates = [
            tf.scatter_update(accumulator, grad_indices, dim_accumulator)
        ]
    with tf.control_dependencies(updates):
      return tf.scatter_sub(var, grad_indices,
                            self._learning_rate_tensor * scaled_g)

  def _resource_apply_dense(self, grad, var):
    return self._apply_dense(grad, var)

  def _resource_apply_sparse(self, grad_values, var, grad_indices):
    return self._apply_sparse_shared(grad_values, grad_indices, var)

  def _apply_sparse(self, grad, var):
    return self._apply_sparse_shared(grad.values, grad.indices, var)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self._weight_decay_rate:
      return False
    if self._exclude_from_weight_decay:
      m = re.match("^(.*):\\d+$", param_name)
      if m is not None:
        param_name = m.group(1)

      for r in self._exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True
