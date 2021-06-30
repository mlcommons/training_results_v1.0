"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow.compat.v1 as tf

from REDACTED.mlperf.submissions.training.v1_0.models.bert import adafactor
from REDACTED.mlperf.submissions.training.v1_0.models.bert import deferred_grad_optimizer
from REDACTED.mlperf.submissions.training.v1_0.models.bert import lamb_optimizer
from REDACTED.mlperf.submissions.training.v1_0.models.bert import sm3
from REDACTED import distributed_matrix_precond


def create_optimizer(grads,
                     init_lr,
                     num_train_steps,
                     num_warmup_steps,
                     use_tpu,
                     optimizer_name="adamw",
                     poly_power=1.0,
                     start_warmup_step=0,
                     weight_decay_rate=0.01,
                     beta_1=0.9,
                     beta_2=0.999,
                     log_epsilon=-6,
                     use_bfloat16_all_reduce=False,
                     steps_per_update=1,
                     clip_by_global_norm_after_grad=False):
  """Creates an optimizer training op."""
  global_step = tf.train.get_or_create_global_step()

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.
  if optimizer_name != "sm3":
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=poly_power,
        cycle=False)

  # Implements linear warmup. I.e., if global_step - start_warmup_step <
  # num_warmup_steps, the learning rate will be
  # `(global_step - start_warmup_step)/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    tf.logging.info("++++++ warmup starts at step " + str(start_warmup_step) +
                    ", for " + str(num_warmup_steps) + " steps ++++++")
    global_steps_int = tf.cast(global_step, tf.int32)
    start_warm_int = tf.constant(start_warmup_step, dtype=tf.int32)
    global_steps_int = global_steps_int - start_warm_int
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    if optimizer_name == "sm3" or optimizer_name == "sm3_beta2":
      warmup_learning_rate = init_lr * tf.pow(warmup_percent_done, 2)
    else:
      warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = ((1.0 - is_warmup) * learning_rate +
                     is_warmup * warmup_learning_rate)
  post_training_loop_callback_fn = None
  # It is OK that you use this optimizer for finetuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  # It is OK to use AdamW in the finetuning even the model is trained by LAMB.
  # As report in the Bert pulic github, the learning rate for SQuAD 1.1 finetune
  # is 3e-5, 4e-5 or 5e-5. For LAMB, the users can use 3e-4, 4e-4,or 5e-4 for a
  # batch size of 64 in the finetune.
  if optimizer_name == "adamw":
    tf.logging.info("using adamw")
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=weight_decay_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=10**(log_epsilon),
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  elif optimizer_name == "lamb":
    tf.logging.info("using lamb")
    optimizer = lamb_optimizer.LAMBOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=weight_decay_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=10**(log_epsilon),
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
        clip_by_global_norm_after_gradient_allreduce=clip_by_global_norm_after_grad
    )
  elif optimizer_name == "adafactor":
    optimizer = adafactor.AdafactorOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=weight_decay_rate,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  elif optimizer_name == "sm3":
    optimizer = sm3.SM3Optimizer(
        learning_rate=learning_rate,
        momentum=beta_1,
        weight_decay_rate=weight_decay_rate,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  elif optimizer_name == "sm3_beta2":
    optimizer = sm3.SM3Optimizer(
        learning_rate=learning_rate,
        momentum=beta_1,
        beta2=beta_2,
        weight_decay_rate=weight_decay_rate,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  elif optimizer_name == "shampoo":
    optimizer = distributed_matrix_precond.MatrixPreconditionerOptimizer(
        learning_rate=learning_rate,
        momentum=beta_1,
        # TODO: Move these to hparams.
        initial_accumulator_value=0.0,
        second_moment_averaging=beta_2,
        bfloat16=True,
        statistics_computation_frequency=1,
        start_preconditioning_steps=1000,
        matrix_epsilon=10**(log_epsilon),
        max_any_dim=8192,
        block_size=1024,
        fallback_to_diagonal_dim=1024,
        global_step=global_step,
        exponent_multiplier=2.0,
        weight_decay_rate=weight_decay_rate,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
        synchronous_preconditioning=False,
        use_weight_norm_multiplier=True)

    def callback_fn():
      invoke_async_ops = optimizer.invoke_async_preconditioner_computation(
          tf.cast(global_step, tf.int32))
      with tf.control_dependencies([invoke_async_ops]):
        return optimizer.assign_preconditioner_to_host_vars()

    post_training_loop_callback_fn = callback_fn
  else:
    raise ValueError("Not supported optimizer: ", optimizer)

  if steps_per_update > 1:
    tf.logging.info("applying gradient aggregation")
    optimizer = deferred_grad_optimizer.GradientAggregationOptimizer(
        optimizer, steps_per_update)

  tvars = tf.trainable_variables()

  # TODO: Try sm3_beta2 with clip by norm.
  if not clip_by_global_norm_after_grad and optimizer_name not in [
      "sm3", "adafactor", "sm3_beta2"
  ]:
    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

  grads_tvars = zip(grads, tvars)
  if use_tpu:
    if use_bfloat16_all_reduce:
      grads_tvars = [(tf.cast(
          tf.tpu.cross_replica_sum(tf.cast(g, tf.bfloat16)), tf.float32), v)
                     for g, v in grads_tvars]
    else:
      grads_tvars = [(tf.tpu.cross_replica_sum(g), v) for g, v in grads_tvars]

  train_op = optimizer.apply_gradients(grads_tvars, global_step=global_step)

  if optimizer_name == "adamw":
    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamWeightDecayOptimizer` does not do this.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op, learning_rate, post_training_loop_callback_fn


class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    b1 = tf.get_variable(
        name="beta_1",
        shape=[],
        dtype=tf.float32,
        trainable=False,
        initializer=tf.constant_initializer(self.beta_1))
    b2 = tf.get_variable(
        name="beta_2",
        shape=[],
        dtype=tf.float32,
        trainable=False,
        initializer=tf.constant_initializer(self.beta_2))
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) +
          tf.multiply(1.0 - self.beta_2, tf.square(grad)))

      update = (next_m / (1.0 - b1)) / (
          tf.sqrt(next_v / (1.0 - b2)) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      update_with_lr = self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    # outside the var loop
    with tf.control_dependencies(assignments):
      new_assignments = [
          b1.assign(b1 * self.beta_1),
          b2.assign(b2 * self.beta_2)
      ]
    return tf.group(*new_assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
