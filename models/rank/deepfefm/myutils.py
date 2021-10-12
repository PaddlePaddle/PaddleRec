# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


def batch_dot(x, y, axes=None):
    """Batchwise dot product.
    # >>> x_batch = paddle.ones(shape=(32, 20, 1))
    # >>> y_batch = paddle.ones(shape=(32, 30, 20))
    # >>> xy_batch_dot = batch_dot(x_batch, y_batch, axes=(1, 2))
    # >>> xy_batch_dot.shape
    (32, 1, 30)

    Shape inference:
    Let `x`'s shape be `(100, 20)` and `y`'s shape be `(100, 30, 20)`.
    If `axes` is (1, 2), to find the output shape of resultant tensor,
        loop through each dimension in `x`'s shape and `y`'s shape:
    * `x.shape[0]` : 100 : append to output shape
    * `x.shape[1]` : 20 : do not append to output shape,
        dimension 1 of `x` has been summed over. (`dot_axes[0]` = 1)
    * `y.shape[0]` : 100 : do not append to output shape,
        always ignore first dimension of `y`
    * `y.shape[1]` : 30 : append to output shape
    * `y.shape[2]` : 20 : do not append to output shape,
        dimension 2 of `y` has been summed over. (`dot_axes[1]` = 2)
    `output_shape` = `(100, 30)`
    """
    x_shape = x.shape
    y_shape = y.shape

    x_ndim = len(x_shape)
    y_ndim = len(y_shape)

    if x_ndim < 2 or y_ndim < 2:
        raise ValueError('Cannot do batch_dot on inputs '
                         'with rank < 2. '
                         'Received inputs with shapes ' + str(x_shape) +
                         ' and ' + str(y_shape) + '.')

    x_batch_size = x_shape[0]
    y_batch_size = y_shape[0]

    if x_batch_size is not None and y_batch_size is not None:
        if x_batch_size != y_batch_size:
            raise ValueError('Cannot do batch_dot on inputs '
                             'with different batch sizes. '
                             'Received inputs with shapes ' + str(x_shape) +
                             ' and ' + str(y_shape) + '.')
    if isinstance(axes, int):
        axes = [axes, axes]

    if axes is None:
        if y_ndim == 2:
            axes = [x_ndim - 1, y_ndim - 1]
        else:
            axes = [x_ndim - 1, y_ndim - 2]

    if any(isinstance(a, (list, tuple)) for a in axes):
        raise ValueError('Multiple target dimensions are not supported. ' +
                         'Expected: None, int, (int, int), ' + 'Provided: ' +
                         str(axes))

    # if tuple, convert to list.
    axes = list(axes)

    # convert negative indices.
    if axes[0] < 0:
        axes[0] += x_ndim
    if axes[1] < 0:
        axes[1] += y_ndim

    # sanity checks
    if 0 in axes:
        raise ValueError('Cannot perform batch_dot over axis 0. '
                         'If your inputs are not batched, '
                         'add a dummy batch dimension to your '
                         'inputs using K.expand_dims(x, 0)')
    a0, a1 = axes
    d1 = x_shape[a0]
    d2 = y_shape[a1]

    if d1 is not None and d2 is not None and d1 != d2:
        raise ValueError('Cannot do batch_dot on inputs with shapes ' + str(
            x_shape) + ' and ' + str(y_shape) + ' with axes=' + str(axes) +
                         '. x.shape[%d] != '
                         'y.shape[%d] (%d != %d).' % (axes[0], axes[1], d1, d2
                                                      ))

    # backup ndims. Need them later.
    orig_x_ndim = x_ndim
    orig_y_ndim = y_ndim

    # if rank is 2, expand to 3.
    if x_ndim == 2:
        x = paddle.unsqueeze(x, 1)
        a0 += 1
        x_ndim += 1
    if y_ndim == 2:
        y = paddle.unsqueeze(y, 2)
        y_ndim += 1

    # bring x's dimension to be reduced to last axis.
    if a0 != x_ndim - 1:
        pattern = list(range(x_ndim))
        for i in range(a0, x_ndim - 1):
            pattern[i] = pattern[i + 1]
        pattern[-1] = a0
        x = paddle.transpose(x, pattern)

    # bring y's dimension to be reduced to axis 1.
    if a1 != 1:
        pattern = list(range(y_ndim))
        for i in range(a1, 1, -1):
            pattern[i] = pattern[i - 1]
        pattern[1] = a1
        y = paddle.transpose(y, pattern)

    # normalize both inputs to rank 3.
    if x_ndim > 3:
        # squash middle dimensions of x.
        x_shape = x.shape
        x_mid_dims = x_shape[1:-1]
        x_squashed_shape = paddle.stack([x_shape[0], -1, x_shape[-1]])
        x = paddle.reshape(x, x_squashed_shape)
        x_squashed = True
    else:
        x_squashed = False

    if y_ndim > 3:
        # squash trailing dimensions of y.
        y_shape = y.shape
        y_trail_dims = y_shape[2:]
        y_squashed_shape = paddle.stack([y_shape[0], y_shape[1], -1])
        y = paddle.reshape(y, y_squashed_shape)
        y_squashed = True
    else:
        y_squashed = False

    result = paddle.matmul(x, y)

    # if inputs were squashed, we have to reshape the matmul output.
    output_shape = paddle.shape(result)
    do_reshape = False

    if x_squashed:
        output_shape = paddle.concat(
            [output_shape[:1], x_mid_dims, output_shape[-1:]], 0)
        do_reshape = True

    if y_squashed:
        output_shape = paddle.concat([output_shape[:-1], y_trail_dims], 0)
        do_reshape = True

    if do_reshape:
        result = paddle.reshape(result, output_shape)

    # if the inputs were originally rank 2, we remove the added 1 dim.
    if orig_x_ndim == 2:
        result = paddle.squeeze(result, 1)
    elif orig_y_ndim == 2:
        result = paddle.squeeze(result, -1)

    return result


if __name__ == '__main__':
    import numpy as np

    # x_batch = paddle.ones(shape=(32, 20, 1))
    # y_batch = paddle.ones(shape=(32, 30, 20))

    x_batch = np.array(
        [[-1.0115546, -0.02948455, 0.871699],
         [0.08505919, -0.849537, 0.43243495],
         [0.87515765, 1.0287786, -0.8976419],
         [1.6105489, 0.7082569, 0.12437075]],
        dtype='float32')
    y_batch = np.array(
        [[-0.8520051, 0.47021824, 0.8739443],
         [-1.1984695, -1.0846833, 0.630532],
         [1.0684944, -1.504634, -0.23854674],
         [-0.7199577, -0.47609442, -0.64525014]],
        dtype='float32')

    x_batch = paddle.to_tensor(x_batch)
    y_batch = paddle.to_tensor(y_batch)

    xy_batch_dot = batch_dot(x_batch, y_batch, axes=1)
    print(xy_batch_dot)
    """
    Tensor(shape=[4, 1], dtype=float32, place=CPUPlace, stop_gradient=True,
       [[ 1.60980189],
        [ 1.09220183],
        [-0.39870456],
        [-1.57697451]])
    """
