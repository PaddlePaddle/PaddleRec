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
    print('Usage: {} <network building filename> [model_dir]\n'.format(this_name))
    print('    example: {} {}'.format(this_name, os.path.join(dirname, 'example.py')))

class ModelBuilder:
    """
    Attributes:
        _save_path: Save path of programs

        def _inference():
            Build inference network(without loss and optimizer)
            **This function is declared in the network_desc_path file, and will be set in initialize()**

            Returns:
                list<Variable>: inputs
                and
                list<Variable>: outputs
            pass

        def _loss_function(*outputs):
            **This function is declared in the network_desc_path file, and will be set in initialize()**
            Args:
                *outputs: the second result of inference()

            Returns:
                Variable: loss
                and
                list<Variable>: labels
            pass
    """

    def initialize(self, network_desc_path, save_path=None):
        """compile the network description module
        Args:
            network_desc_path: path
            save_path: model save path, default is ./model/<network_desc_path without .py>/

        Returns:
            bool: True if succeed else False
        """
        if not isinstance(network_desc_path, str):
            print('network_desc_path must be str')
            return False

        if not network_desc_path.endswith('.py'):
            print('network_desc_path must be end with .py')
            return False

        if not os.path.exists(network_desc_path):
            print('file not exists:', network_desc_path)
            return False

        scope = dict()
        with open(network_desc_path, 'r') as f:
            code = f.read()
            compiled = compile(code, network_desc_path, 'exec')
            exec(compiled, scope)

        if not 'inference' in scope:
            print('inference not defined')
            return False

        if not 'loss_function' in scope:
            print('loss_function not defined')
            return False

        if save_path is None:
            # example /a/b/c.d -> ./model/c
            save_path = os.path.join('./model', os.path.splitext(os.path.split(network_desc_path)[1])[0])
            print('save in the default path:', save_path)

        self._save_path = save_path

        self._inference = scope['inference']
        self._loss_function = scope['loss_function']

        return True

    def build_and_save(self):
        """Build programs and save to _save_path
        """
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            #input_accessor, sparses, inputs, outputs, monitors 
            inference_info = self._inference()
            inputs = inference_info['inputs']            
            outputs = inference_info['outputs']            
            sparses = inference_info['sparses']            
            monitors = inference_info['monitors']            
            input_accessor = inference_info['accessors']            

            test_program = main_program.clone(for_test=True)
            loss, labels = self._loss_function(*outputs)

            optimizer = fluid.optimizer.SGD(learning_rate=1.0)
            params_grads = optimizer.backward(loss)

        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)

        programs = {
            'startup_program': startup_program,
            'main_program': main_program,
            'test_program': test_program,
        }
        for name, program in programs.items():
            with open(os.path.join(self._save_path, name), 'w') as f:
                f.write(program.desc.serialize_to_string())

        fluid.io.save_inference_model(self._save_path,
                [var.name for var in inputs],
                outputs,
                executor=None,
                main_program=test_program,
                model_filename='inference_program',
                program_only=True)

        params = filter(fluid.io.is_parameter, main_program.list_vars())
        vars = []
        sums=[]
        for param in params:
            if param.name.find("bn") == 0:
                sums.append({"name": param.name, "shape": param.shape});
            else:
                vars.append({"name": param.name, "shape": param.shape});

        for accessor in input_accessor:
            if (accessor["input"] == "sparses"):
                accessor["input"] = sparses
            if (accessor["input"] == "vars"):
                accessor["input"] = vars
            if (accessor["input"] == "sums"):
                accessor["input"] = sums
            if (accessor["input"] == "labels"):
                accessor["input"] = [
                    {"label_name": label.name, "shape": label.shape, "output_name": output.name } 
                    for (label, output) in zip(labels, outputs) ]
       
        for monitor in monitors:
            idx = outputs.index(monitor['target'])
            monitor["target_idx"] = idx
            monitor["target"] = outputs[idx].name

        model_desc_path = os.path.join(self._save_path, 'model.yaml')
        model_desc = {
            'inputs': [{"name": var.name, "shape": var.shape} for var in inputs],
            'outputs': [{"name": var.name, "shape": var.shape} for var in outputs],
            'labels': [{"name": var.name, "shape": var.shape} for var in labels],
            'loss': loss.name,
            'input_accessor': input_accessor,
            'monitor': monitors,
            'aa_Attention': 'Do Not Modify This File Manually, Unless You Really Know It'
        }

        with open(model_desc_path, 'w') as f:
            yaml.safe_dump(model_desc, f, encoding='utf-8', allow_unicode=True, default_flow_style=None)


def main(argv):
    """Create programs
    Args:
        argv: arg list, length should be 2
    """
    if len(argv) < 2:
        print_help(argv[0])
        exit(1)
    network_desc_path = argv[1]

    if len(argv) > 2:
        save_path = argv[2]
    else:
        save_path = None

    builder = ModelBuilder()
    if not builder.initialize(network_desc_path, save_path):
        print_help(argv[0])
        exit(1)
    builder.build_and_save()

if __name__ == "__main__":
    main(sys.argv)
