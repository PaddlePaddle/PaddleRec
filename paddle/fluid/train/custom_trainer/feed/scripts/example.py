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
        Variable: ctr_output
    """
    # TODO: build network here
    cvm_input = fluid.layers.data(name='cvm_input', shape=[4488], dtype='float32')

    net = cvm_input
    net = fluid.layers.fc(net, 512, act='relu')
    net = fluid.layers.fc(net, 256, act='relu')
    net = fluid.layers.fc(net, 256, act='relu')
    net = fluid.layers.fc(net, 128, act='relu')
    net = fluid.layers.fc(net, 128, act='relu')
    net = fluid.layers.fc(net, 128, act='relu')
    net = fluid.layers.fc(net, 128, act='relu')

    ctr_output = fluid.layers.fc(net, 1, act='sigmoid', name='ctr_output')
    return [cvm_input], ctr_output
