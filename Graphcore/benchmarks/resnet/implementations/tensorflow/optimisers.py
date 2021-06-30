# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import math_ops

class IPU_Momentum(tf.train.MomentumOptimizer):
    def __init__(self, learning_rate, momentum,
                 momentum_dtype=None, gradient_dtype=None,
                 name="Momentum"):
        super(IPU_Momentum, self).__init__(False, name)
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._momentum_dtype = momentum_dtype
        self._gradient_dtype = gradient_dtype
        self._use_nesterov = False

    def _create_slots(self, var_list):
        for v in var_list:
            if self._momentum_dtype==None:
                m_dtype = v.dtype.base_dtype
            else:
                m_dtype = self._momentum_dtype
            self._get_or_make_slot_with_initializer(v,
                                                    tf.zeros_initializer(),
                                                    v.get_shape(),
                                                    m_dtype,
                                                    "momentum", self._name)

    def compute_gradients(self, *args, **kwargs):  # pylint: disable=arguments-differ
        grads_and_vars = super().compute_gradients(*args, **kwargs)
        if self._gradient_dtype == None:
            return grads_and_vars
        else:
            return [(tf.cast(g, self._gradient_dtype), v)
                    for g, v in grads_and_vars]

    def _resource_apply_dense(self, grad, var):
        mom = self.get_slot(var, "momentum")
        grad = math_ops.cast(grad, tf.float32)
        lr = math_ops.cast(self._learning_rate_tensor, tf.float32)
        m = math_ops.cast(self._momentum_tensor, tf.float32)
        var32 = math_ops.cast(var, tf.float32)
        mom32 = math_ops.cast(mom, tf.float32)
        mom_up = m * mom32 + grad
        var_up = var32 - (lr * mom_up)
        return tf.group(var.assign(tf.cast(var_up, var.dtype.base_dtype)),
                        mom.assign(tf.cast(mom_up, mom.dtype.base_dtype)))


class IPU_LARS(IPU_Momentum):
    def __init__(self,
                 learning_rate,
                 momentum,
                 weight_decay=0.0,
                 eta=0.001,
                 epsilon=0.0,
                 exclude_filter=None,
                 momentum_dtype=None,
                 gradient_dtype=None,
                 name="LARS"):
        super(IPU_LARS, self).__init__(learning_rate,
                                       momentum,
                                       momentum_dtype,
                                       gradient_dtype,
                                       name)
        self._weight_decay = weight_decay
        self._eta = eta
        self._epsilon = epsilon
        self._exclude_filter = [] if exclude_filter is None else exclude_filter

    def _resource_apply_dense(self, grad, var):
        mom = self.get_slot(var, "momentum")
        grad = math_ops.cast(grad, tf.float32)
        lr = math_ops.cast(self._learning_rate_tensor, tf.float32)
        m = math_ops.cast(self._momentum_tensor, tf.float32)
        var32 = math_ops.cast(var, tf.float32)
        mom32 = math_ops.cast(mom, tf.float32)

        if not any(exclude in var.name for exclude in self._exclude_filter):
            w_norm = tf.norm(var32, ord=2)
            g_norm = tf.norm(grad, ord=2)
            trust_ratio = (self._eta * w_norm /
                     (g_norm + self._weight_decay * w_norm + self._epsilon))
            trust_ratio = tf.where(
                tf.logical_and(tf.greater(w_norm, 0), tf.greater(g_norm, 0)),
                trust_ratio, 1.0)
            lr = lr * trust_ratio

            grad = grad + self._weight_decay * var

        mom_up = m * mom32 + (lr * grad)
        var_up = var32 - mom_up
        return tf.group(var.assign(tf.cast(var_up, var.dtype.base_dtype)),
                        mom.assign(tf.cast(mom_up, mom.dtype.base_dtype)))