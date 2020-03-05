import copy
import yaml
import kagle_layer
import kagle_table
import paddle.fluid as fluid
from abc import ABCMeta, abstractmethod
from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet

def create(config):
    model = None
    if config['mode'] == 'fluid':
        model = FluidModel(config)
        model.build_model()
    return model
   
class Model(object):
    __metaclass__=ABCMeta

    def __init__(self, config):
        self._config = config
        self._name = config['name']
        f = open(config['layer_file'], 'r')
        self._build_nodes = yaml.safe_load(f.read())
        self._build_phase = ['input', 'param', 'summary', 'layer']
        self._build_param = {'layer': {}, 'inner_layer':{}, 'layer_extend': {}, 'model': {}}
        self._inference_meta = {'dependency':{}, 'params': {}}
        self._cost = None
        self._metrics = {}
        self._data_var = []
        pass
    
    def get_cost_op(self):
        return self._cost

    def get_metrics(self):
        return self._metrics

    @abstractmethod
    def shrink(self, params):
        pass    

    @abstractmethod
    def build_model(self): 
        pass

    @abstractmethod
    def dump_model_program(self, path):
        pass

    @abstractmethod
    def dump_inference_param(self, params):
        pass
    @abstractmethod
    def dump_inference_program(self, inference_layer, path):
        pass
    
    def inference_params(self, inference_layer):
        layer = inference_layer
        if layer in self._inference_meta['params']:
            return self._inference_meta['params'][layer]
            
        self._inference_meta['params'][layer] = []
        self._inference_meta['dependency'][layer] = self.get_dependency(self._build_param['inner_layer'], layer)
        for node in self._build_nodes['layer']:
            if node['name'] not in self._inference_meta['dependency'][layer]:
                continue
            if 'inference_param' in self._build_param['layer_extend'][node['name']]:
                self._inference_meta['params'][layer] += self._build_param['layer_extend'][node['name']]['inference_param']['params'] 
        return self._inference_meta['params'][layer]

    def get_dependency(self, layer_graph, dest_layer):
        dependency_list = []
        if dest_layer in layer_graph:
            dependencys = copy.deepcopy(layer_graph[dest_layer]['input'])
            dependency_list = copy.deepcopy(dependencys)
            for dependency in dependencys:
                dependency_list = dependency_list + self.get_dependency(layer_graph, dependency)
        return list(set(dependency_list))

    
class FluidModel(Model):
    def __init__(self, config):
        Model.__init__(self, config)
        pass
    
    def build_model(self): 
        for layer in self._build_nodes['layer']:
            self._build_param['inner_layer'][layer['name']] = layer
        
    
        self._build_param['table'] = {}
        self._build_param['model']['train_program'] = fluid.Program()
        self._build_param['model']['startup_program'] = fluid.Program()
        with fluid.program_guard(self._build_param['model']['train_program'], self._build_param['model']['startup_program']):
            with fluid.unique_name.guard():
                for phase in self._build_phase:
                    if self._build_nodes[phase] is None:
                        continue
                    for node in self._build_nodes[phase]:
                        exec("""layer=kagle_layer.{}(node)""".format(node['class']))
                        layer_output, extend_output = layer.generate(self._config['mode'], self._build_param)
                        self._build_param['layer'][node['name']] = layer_output
                        self._build_param['layer_extend'][node['name']] = extend_output
                        if extend_output is None:
                            continue
                        if 'loss' in extend_output:
                            if self._cost is None:
                                self._cost = extend_output['loss']
                            else:
                                self._cost += extend_output['loss']
                        if 'data_var' in extend_output:
                            self._data_var += extend_output['data_var']
                        if 'metric_label' in extend_output and extend_output['metric_label'] is not None:
                            self._metrics[extend_output['metric_label']] = extend_output['metric_dict']

                        if 'inference_param' in extend_output:
                            param_name = extend_output['inference_param']['name']
                            if param_name not in self._build_param['table']:
                                self._build_param['table'][param_name] = {'params':[]}
                                table_meta = kagle_table.TableMeta.alloc_new_table(extend_output['inference_param']['table_id'])
                                self._build_param['table'][param_name]['_meta'] = table_meta
                            self._build_param['table'][param_name]['params'] += extend_output['inference_param']['params']
        pass
    
    @classmethod
    def build_optimizer(self, params):
        optimizer_conf = params['optimizer_conf']
        strategy = None
        if 'strategy' in optimizer_conf:
            strategy = optimizer_conf['strategy']
            stat_var_names = []
            metrics = params['metrics']
            for name in metrics:
                model_metrics = metrics[name]
                stat_var_names += [ model_metrics[metric]['var'].name for metric in model_metrics]
            strategy['stat_var_names'] = list(set(stat_var_names))
        optimizer_generator = 'optimizer = fluid.optimizer.' + optimizer_conf['class'] + '(learning_rate=' +  str(optimizer_conf['learning_rate']) + ')'
        exec(optimizer_generator)            
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        return optimizer

    def dump_model_program(self, path):
        with open(path + '/' + self._name + '_main_program.pbtxt', "w") as fout:
            print >> fout, self._build_param['model']['train_program']
        with open(path + '/' + self._name + '_startup_program.pbtxt', "w") as fout:
            print >> fout, self._build_param['model']['startup_program']
        pass

    def shrink(self, params):
        scope = params['scope']
        decay = params['decay']
        for param_table in self._build_param['table']:
            table_id = self._build_param['table'][param_table]['_meta']._table_id
            fleet.shrink_dense_table(decay, scope=scope, table_id=table_id)

    def dump_inference_program(self, inference_layer, path):
        pass

    def dump_inference_param(self, params):
        scope = params['scope']
        executor = params['executor']
        program = self._build_param['model']['train_program']
        for table_name,table in self._build_param['table'].items():
            fleet._fleet_ptr.pull_dense(scope, table['_meta']._table_id, table['params'])
        for infernce_item in params['inference_list']:
            params_name_list = self.inference_params(infernce_item['layer_name'])
            params_var_list = [ program.global_block().var(i) for i in params_name_list ]
            params_file_name = infernce_item['save_file_name']
            with fluid.scope_guard(scope):
                if params['save_combine']:
                    fluid.io.save_vars(
                    executor, "./", program, vars=params_var_list, filename=params_file_name)
                else:
                    fluid.io.save_vars(executor, params_file_name, program, vars=params_var_list)
        pass
