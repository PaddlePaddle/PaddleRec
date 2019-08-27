#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
This is an example of network building
"""

from __future__ import print_function, division
import paddle
from paddle import fluid

def inference():
    """Build inference network(without loss and optimizer)

    Returns:
        list<Variable>: inputs
        and
        list<Variable>: outputs
    """
    # TODO: build network here
    cvm_input = fluid.layers.data(name='cvm_input', shape=[4488], dtype='float32', stop_gradient=False)

    net = cvm_input
    net = fluid.layers.data_norm(input=net, name="bn6048", epsilon=1e-4,
        param_attr={"batch_size":1e4, "batch_sum_default":0.0, "batch_square":1e4})
    net = fluid.layers.fc(net, 512, act='relu', name='fc_1')
    net = fluid.layers.fc(net, 256, act='relu', name='fc_2')
    net = fluid.layers.fc(net, 256, act='relu', name='fc_3')
    net = fluid.layers.fc(net, 128, act='relu', name='fc_4')
    net = fluid.layers.fc(net, 128, act='relu', name='fc_5')
    net = fluid.layers.fc(net, 128, act='relu', name='fc_6')
    net = fluid.layers.fc(net, 128, act='relu', name='fc_7')

    ctr_output = fluid.layers.fc(net, 1, act='sigmoid', name='ctr')
    return [cvm_input], [ctr_output]

def loss_function(ctr_output):
    """
    Args:
        *outputs: the second result of inference()

    Returns:
        Variable: loss
        and
        list<Variable>: labels
    """
    # TODO: calc loss here

    label = fluid.layers.data(name='label_ctr', shape=ctr_output.shape, dtype='float32')
    loss = fluid.layers.square_error_cost(input=ctr_output, label=label)
    loss = fluid.layers.mean(loss, name='loss_ctr')

    return loss, [label]
