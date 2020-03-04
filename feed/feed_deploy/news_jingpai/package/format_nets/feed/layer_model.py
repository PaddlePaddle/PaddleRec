import os
import copy
import yaml
import layer_model
import paddle.fluid as fluid

mode='fluid'
f = open('model.layers', 'r')


build_nodes = yaml.safe_load(f.read())


build_param = {'layer': {}, 'inner_layer':{}, 'layer_extend': {}, 'model': {}}
build_phase = ['input', 'param', 'layer']
inference_layer = ['ctr_output']
inference_meta = {'dependency':{}, 'params': {}}
for layer in build_nodes['layer']:
    build_param['inner_layer'][layer['name']] = layer

def get_dependency(layer_graph, dest_layer):
    dependency_list = []
    if dest_layer in layer_graph:
        dependencys = copy.deepcopy(layer_graph[dest_layer]['input'])
        dependency_list = copy.deepcopy(dependencys)
        for dependency in dependencys:
            dependency_list = dependency_list + get_dependency(layer_graph, dependency)
    return list(set(dependency_list))

# build train model
if mode == 'fluid':
    build_param['model']['train_program'] = fluid.Program()
    build_param['model']['startup_program'] = fluid.Program()
    with fluid.program_guard(build_param['model']['train_program'], build_param['model']['startup_program']):
        with fluid.unique_name.guard():
            for phase in build_phase:
                for node in build_nodes[phase]:
                    exec("""layer=layer_model.{}(node)""".format(node['class']))
                    layer_output, extend_output = layer.generate(mode, build_param)
                    build_param['layer'][node['name']] = layer_output
                    build_param['layer_extend'][node['name']] = extend_output

# build inference model
for layer in inference_layer:
    inference_meta['param'][layer] = []
    inference_meta['dependency'][layer] = get_dependency(build_param['inner_layer'], layer)
    for node in build_nodes['layer']:
        if node['name'] not in inference_meta['dependency'][layer]:
            continue
        if 'inference_param' in build_param['layer_extend'][node['name']]:
            inference_meta['param'][layer] += build_param['layer_extend'][node['name']]['inference_param'] 
    print(inference_meta['param'][layer])


