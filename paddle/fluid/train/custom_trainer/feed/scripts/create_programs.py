#!/usr/bin/env python
#-*- coding:utf-8 -*-

from __future__ import print_function, division
import os
import sys
import paddle
from paddle import fluid
import yaml

def print_help(this_name):
    """Print help
    """
    dirname = os.path.dirname(this_name)
    print("Usage: {} <network building filename> [model_dir]\n".format(this_name))
    print("    example: {} {}".format(this_name, os.path.join(dirname, 'example.py')))

def inference(filename):
    """Build inference network(without loss and optimizer)
    Args:
        filename: path of file which defined real inference function
    Returns:
        list<Variable>: inputs
        and
        Variable: ctr_output
    """
    with open(filename, 'r') as f:
        code = f.read()
    compiled = compile(code, filename, 'exec')
    exec(compiled)
    return inference()

def main(argv):
    """Create programs
    Args:
        argv: arg list, length should be 2
    """
    if len(argv) < 2 or not os.path.exists(argv[1]):
        print_help(argv[0])
        exit(1)
    network_build_file = argv[1]

    if len(argv) >= 2:
        model_dir = argv[2]
    else:
        model_dir = './model'

    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        inputs, ctr_output = inference(network_build_file)

        test_program = main_program.clone(for_test=True)

        label_target = fluid.layers.data(name='label', shape=[1], dtype='float32')

        loss = fluid.layers.square_error_cost(input=ctr_output, label=label_target)
        loss = fluid.layers.mean(loss, name='loss')
        
        optimizer = fluid.optimizer.SGD(learning_rate=1.0)
        params_grads = optimizer.backward(loss)

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    programs = {
        'startup_program': startup_program,
        'main_program': main_program,
        'test_program': test_program,
    }
    for save_path, program in programs.items():
        with open(os.path.join(model_dir, save_path), 'w') as f:
            f.write(program.desc.serialize_to_string())

    model_desc_path = os.path.join(model_dir, 'model.yaml')
    model_desc = dict()
    model_desc['inputs'] = {var.name: var.shape for var in inputs}
    model_desc['loss_name'] = loss.name
    model_desc['label_name'] = label_target.name
    model_desc['ctr_output_name'] = ctr_output.name
    
    with open(model_desc_path, 'w') as f:
        yaml.safe_dump(model_desc, f, encoding='utf-8', allow_unicode=True)


if __name__ == "__main__":
    main(sys.argv)
