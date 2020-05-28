# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""A library of helpers for use with SamplingDecoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.distributions import bernoulli
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.util import nest

__all__ = [
    "Helper",
    "InferenceHelper",
]

_transpose_batch_time = decoder._transpose_batch_time  # pylint: disable=protected-access


def _unstack_ta(inp):
  return tensor_array_ops.TensorArray(
      dtype=inp.dtype, size=array_ops.shape(inp)[0],
      element_shape=inp.get_shape()[1:]).unstack(inp)


@six.add_metaclass(abc.ABCMeta)
class Helper(object):
  """Interface for implementing sampling in seq2seq decoders.
  Helper instances are used by `BasicDecoder`.
  """

  @abc.abstractproperty
  def batch_size(self):
    """Batch size of tensor returned by `sample`.
    Returns a scalar int32 tensor.
    """
    raise NotImplementedError("batch_size has not been implemented")

  @abc.abstractproperty
  def sample_ids_shape(self):
    """Shape of tensor returned by `sample`, excluding the batch dimension.
    Returns a `TensorShape`.
    """
    raise NotImplementedError("sample_ids_shape has not been implemented")

  @abc.abstractproperty
  def sample_ids_dtype(self):
    """DType of tensor returned by `sample`.
    Returns a DType.
    """
    raise NotImplementedError("sample_ids_dtype has not been implemented")

  @abc.abstractmethod
  def initialize(self, name=None):
    """Returns `(initial_finished, initial_inputs)`."""
    pass

  @abc.abstractmethod
  def sample(self, time, outputs, state, name=None):
    """Returns `sample_ids`."""
    pass

  @abc.abstractmethod
  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    """Returns `(finished, next_inputs, next_state)`."""
    pass

class InferenceHelper(Helper):
  """A helper to use during inference with a custom sampling function."""

  def __init__(self, sample_fn, sample_shape, sample_dtype,
               start_inputs, end_fn, next_inputs_fn=None):
    """Initializer.
    Args:
      sample_fn: A callable that takes `outputs` and emits tensor `sample_ids`.
      sample_shape: Either a list of integers, or a 1-D Tensor of type `int32`,
        the shape of the each sample in the batch returned by `sample_fn`.
      sample_dtype: the dtype of the sample returned by `sample_fn`.
      start_inputs: The initial batch of inputs.
      end_fn: A callable that takes `sample_ids` and emits a `bool` vector
        shaped `[batch_size]` indicating whether each sample is an end token.
      next_inputs_fn: (Optional) A callable that takes `sample_ids` and returns
        the next batch of inputs. If not provided, `sample_ids` is used as the
        next batch of inputs.
    """
    self._sample_fn = sample_fn
    self._end_fn = end_fn
    self._sample_shape = tensor_shape.TensorShape(sample_shape)
    self._sample_dtype = sample_dtype
    self._next_inputs_fn = next_inputs_fn
    self._batch_size = start_inputs.shape[0]#array_ops.shape(start_inputs)[0]
    self._start_inputs = ops.convert_to_tensor(
        start_inputs, name="start_inputs")

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def sample_ids_shape(self):
    return self._sample_shape

  @property
  def sample_ids_dtype(self):
    return self._sample_dtype

  def initialize(self, name=None):
    finished = array_ops.tile([False], [self._batch_size])
    print('finished',finished,'and shape', finished.shape)
    return (finished, self._start_inputs)

  def sample(self, time, outputs, state, name=None):
    del time, state  # unused by sample
    return self._sample_fn(outputs)

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    del time, outputs  # unused by next_inputs
    if self._next_inputs_fn is None:
      next_inputs = sample_ids
    else:
      next_inputs = self._next_inputs_fn(sample_ids)
    finished = self._end_fn(sample_ids)
    return (finished, next_inputs, state)