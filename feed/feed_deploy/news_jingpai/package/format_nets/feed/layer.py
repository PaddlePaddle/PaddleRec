import paddle.fluid as fluid
from abc import ABCMeta, abstractmethod

class Layer(object):
    __metaclass__=ABCMeta

    def __init__(self, config):
        pass
    
    def generate(self, mode, param): 
        if mode == 'fluid':
            return self.generate_fluid(param)
        elif mode == 'tensorflow':
            return self.generate_tensorflow(param)
        print ('unsupport this mode: ' + mode) 
        return None,None

    @abstractmethod
    def generate_fluid(self, param): 
        pass

    @abstractmethod
    def generate_tensorflow(self, param): 
        pass

class EmbeddingInputLayer(Layer):
    def __init__(self, config):
        self._cvm = config['cvm']
        self._name = config['name']
        self._slots = config['slots']
        self._mf_dim = config['mf_dim']
        self._backward = config['backward']
        self._emb_dim = self._mf_dim
        if self._cvm:
            self._emb_dim = self._mf_dim + 2 #append show ctr
        self._emb_layers = []
    
    def generate_fluid(self, param): 
        show_clk = fluid.layers.concat(
            [param['layer']['show'], param['layer']['click']], axis=1)
        show_clk.stop_gradient = True
        for slot in self._slots:
            l = fluid.layers.data(name=slot, shape=[1], dtype="int64", lod_level=1)
            emb = fluid.layers.embedding(input=l, size=[10, self._mf_dim + 2], is_sparse = True, is_distributed=True, param_attr=fluid.ParamAttr(name="embedding"))
            emb = fluid.layers.sequence_pool(input=emb, pool_type='sum')
            emb = fluid.layers.continuous_value_model(emb, show_clk, self._use_cvm)
            self._emb_layers.append(emb)
        output = fluid.layers.concat(input=self._emb_layers, axis=1, name=self._name)
        return output, None

class LabelInputLayer(Layer):
    def __init__(self, config):
        self._name = config['name']
        self._dim = config.get('dim', 1)
        self._data_type = config.get('data_type', "int64")
        self._label_idx = config['label_idx']

    def generate_fluid(self, param): 
        output = fluid.layers.data(name=self._name, shape=[-1, self._dim], dtype=self._data_type, lod_level=0, append_batch_size=False)
        return output, None

class TagInputLayer(Layer): 
    def __init__(self, config):
        self._name = config['name']
        self._tag = config['tag']
        self._dim = config.get('dim', 1)
        self._data_type = config['data_type']

    def generate_fluid(self, param): 
        output = fluid.layers.data(name=self._name, shape=[-1, self._dim], dtype=self._data_type, lod_level=0, append_batch_size=False, stop_gradient=Tru)
        return output, None

class ParamLayer(Layer): 
    def __init__(self, config):
        self._name = config['name']
        self._coln = config['coln']
        self._init_range = config.get('init_range', 1)
        self._data_type = config['data_type']
        self._config = config

    def generate_fluid(self, param): 
        return config, None

class NormalizetionLayer(Layer): 
    def __init__(self, config):
        self._name = config['name']
        self._input = config['input']

    def generate_fluid(self, param): 
        input_layer = param['layer'][self._input[0]]
        if len(self._input) > 0:
            input_list=[ param['layer'][i] for i in self._input ]
            input_layer = fluid.layers.concat(input=input_list, axis=1)
        bn = fluid.layers.data_norm(input=input_layer, name=self._name, epsilon=1e-4, param_attr={
             "batch_size":1e4,
             "batch_sum_default":0.0,
             "batch_square":1e4})
        inference_param = [ self._name + '.batch_size',  self._name + '.batch_sum',  self._name + '.batch_square_sum' ]
        return bn, {'inference_param' : inference_param}

class NeuralLayer(Layer): 
    def __init__(self, config):
        self._name = config['name']
        self._param = config['param']
        self._input = config['input']
        self._bias = config.get('bias', True)
        self._act_func = config.get('act_func', None)

    def generate_fluid(self, param): 
        param_layer = param['layer'][self._param]
        input_layer = param['layer'][slef._input[0]]
        if len(self._input) > 0:
            input_list=[ param['layer'][i] for i in self._input ]
            input_layer = fluid.layers.concat(input=input_list, axis=1)
        input_coln = input_layer.shape[1]
        scale = param_layer['init_range'] / (input_coln ** 0.5)
        bias = None
        if self._bias:
            bias = fluid.ParamAttr(learning_rate=1.0, initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=scale))
        fc = fluid.layers.fc(
            name = slef._name,
            input = input_layer,
            size = param_layer['coln'],
            act = self._act_func,
            param_attr = \
                fluid.ParamAttr(learning_rate=1.0, \
                initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=scale)),
            bias_attr = bias)
        inference_param = [self._name + '.w_0',  self._name + '.b_0']
        return fc, {'inference_param' : inference_param}

class SigmoidLossLayer(Layer):
    def __init__(self, config):
        self._name = config['name']
        self._label = config['label']
        self._input = config['input']
        self._weight = config.get('weight', None)
        self._bound = config.get('bound', [-15.0, 15.0])
        self._extend_output = {}

    def generate_fluid(self, param): 
        input_layer = param['layer'][slef._input[0]]
        label_layer = param['layer'][slef._label]
        output = fluid.layers.clip(input_layer, min=self._bound[0], max=self._bound[1]), name = self._name)
        norm = fluid.layers.sigmoid(input=output, name=self._name)
        output = fluid.layers.log_loss(input=norm, label=label_layer)
        if self._weight:
            weight_layer = param['layer'][slef._weight]
            output = fluid.layers.elementwise_mul(output, weight_layer)
        output = fluid.layers.mean(x=output)
        
        #For AUC
        binary_predict = fluid.layers.concat(
            input=[fluid.layers.elementwise_sub(fluid.layers.ceil(norm), norm), norm], axis=1)
        self._extend_output['auc'], self._extend_output['batch_auc', [self._extend_output['batch_stat_pos'], \
            self._extend_output['batch_stat_neg'], self._extend_output['stat_pos', self._extend_output['stat_neg']] = \
            fluid.layers.auc(input=binary_predict, label=label_layer, curve='ROC', num_thresholds=4096)

        self._extend_output['sqrerr'], self._extend_output['abserr'], self._extend_output['prob'], self._extend_output['q'], \
            self._extend_output['pos'], self._extend_output['total'] = \
            fluid.contrib.layers.ctr_metric_bundle(norm, fluid.layers.cast(x=label_layer, dtype='float32'))

        return norm, self._extend_output
