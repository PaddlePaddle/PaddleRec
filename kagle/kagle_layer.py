"""
DnnLayer: analyse layer config, and parse to Paddle Operator, build net
"""
import abc
import paddle.fluid as fluid

class Layer(object):
    """R
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, config):
        """R
        """
        pass
    
    def generate(self, mode, param): 
        """R
        """
        if mode == 'fluid':
            return self.generate_fluid(param)
        elif mode == 'tensorflow':
            return self.generate_tensorflow(param)
        print ('unsupport this mode: ' + mode) 
        return None, None

    @abc.abstractmethod
    def generate_fluid(self, param): 
        """R
        """
        pass

    def generate_tensorflow(self, param): 
        """ Not implement currently
        """
        pass


class EmbeddingInputLayer(Layer):
    """R
    """
    def __init__(self, config):
        """R
        """
        self._cvm = config['cvm']
        self._name = config['name']
        self._slots = [str(slot) for slot in config['slots']]
        self._mf_dim = config['mf_dim']
        self._backward = config['backward']
        self._emb_dim = self._mf_dim + 3 #append show ctr lr
        self._emb_layers = []
    
    def generate_fluid(self, param): 
        """R
        """
        show_clk = fluid.layers.concat(
            [param['layer']['show'], param['layer']['click']], axis=1)
        show_clk.stop_gradient = True
        data_var = []
        for slot in self._slots:
            l = fluid.layers.data(name=slot, shape=[1], dtype="int64", lod_level=1)
            data_var.append(l)
            emb = fluid.layers.embedding(input=l, size=[10, self._emb_dim], \
                is_sparse=True, is_distributed=True, param_attr=fluid.ParamAttr(name="embedding"))
            emb = fluid.layers.sequence_pool(input=emb, pool_type='sum')
            emb = fluid.layers.continuous_value_model(emb, show_clk, self._cvm)
            self._emb_layers.append(emb)
        output = fluid.layers.concat(input=self._emb_layers, axis=1, name=self._name)
        return output, {'data_var' : data_var}


class LabelInputLayer(Layer):
    """R
    """
    def __init__(self, config):
        """R
        """
        self._name = config['name']
        self._dim = config.get('dim', 1)
        self._data_type = config.get('data_type', "int64")
        self._label_idx = config['label_idx']

    def generate_fluid(self, param): 
        """R
        """
        label = fluid.layers.data(name=self._name, shape=[-1, self._dim], \
            dtype=self._data_type, lod_level=0, append_batch_size=False)
        cast_label = fluid.layers.cast(label, dtype='float32')
        cast_label.stop_gradient = True
        return cast_label, {'data_var': [label]}


class TagInputLayer(Layer): 
    """R
    """
    def __init__(self, config):
        """R
        """
        self._name = config['name']
        self._tag = config['tag']
        self._dim = config.get('dim', 1)
        self._data_type = config['data_type']

    def generate_fluid(self, param): 
        """R
        """
        output = fluid.layers.data(name=self._name, shape=[-1, self._dim], \
            dtype=self._data_type, lod_level=0, append_batch_size=False, stop_gradient=True)
        return output, {'data_var': [output]}
        

class ParamLayer(Layer): 
    """R
    """
    def __init__(self, config):
        """R
        """
        self._name = config['name']
        self._coln = config['coln']
        self._table_id = config.get('table_id', -1)
        self._init_range = config.get('init_range', 1)
        self._data_type = config.get('data_type', 'float32')
        self._config = config

    def generate_fluid(self, param): 
        """R
        """
        return self._config, {'inference_param': {'name':'param', 'params': [], 'table_id': self._table_id}} 


class SummaryLayer(Layer): 
    """R
    """
    def __init__(self, config):
        """R
        """
        self._name = config['name']
        self._table_id = config.get('table_id', -1)
        self._data_type = config.get('data_type', 'float32')
        self._config = config

    def generate_fluid(self, param): 
        """R
        """
        return self._config, {'inference_param': {'name': 'summary', 'params': [], 'table_id': self._table_id}} 


class NormalizetionLayer(Layer): 
    """R
    """
    def __init__(self, config):
        """R
        """
        self._name = config['name']
        self._input = config['input']
        self._summary = config['summary']       
        self._table_id = config.get('table_id', -1)

    def generate_fluid(self, param): 
        """R
        """
        input_layer = param['layer'][self._input[0]]
        summary_layer = param['layer'][self._summary]
        if len(self._input) > 0:
            input_list=[param['layer'][i] for i in self._input]
            input_layer = fluid.layers.concat(input=input_list, axis=1)
        bn = fluid.layers.data_norm(input=input_layer, name=self._name, epsilon=1e-4, param_attr={
             "batch_size": 1e4, "batch_sum_default": 0.0, "batch_square": 1e4})
        inference_param = [self._name + '.batch_size', self._name + '.batch_sum', self._name + '.batch_square_sum']
        return bn, {'inference_param' : {'name':'summary', 'params': inference_param, 'table_id': summary_layer.get('table_id', -1)}}


class NeuralLayer(Layer): 
    """R
    """
    def __init__(self, config):
        """R
        """
        self._name = config['name']
        self._param = config['param']
        self._input = config['input']
        self._bias = config.get('bias', True)
        self._act_func = config.get('act_func', None)

    def generate_fluid(self, param): 
        """R
        """
        param_layer = param['layer'][self._param]
        input_layer = param['layer'][self._input[0]]
        if len(self._input) > 0:
            input_list=[param['layer'][i] for i in self._input]
            input_layer = fluid.layers.concat(input=input_list, axis=1)
        input_coln = input_layer.shape[1]
        scale = param_layer['init_range'] / (input_coln ** 0.5)
        bias = None
        if self._bias:
            bias = fluid.ParamAttr(learning_rate=1.0, 
                initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=scale))
        fc = fluid.layers.fc(
            name = self._name,
            input = input_layer,
            size = param_layer['coln'],
            act = self._act_func,
            param_attr = \
                fluid.ParamAttr(learning_rate=1.0, \
                initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=scale)),
            bias_attr = bias)
        inference_param = [self._name + '.w_0',  self._name + '.b_0']
        return fc, {'inference_param' : {'name':'param', 'params': inference_param, 'table_id': param_layer.get('table_id', -1)}}


class SigmoidLossLayer(Layer):
    """R
    """
    def __init__(self, config):
        """R
        """
        self._name = config['name']
        self._label = config['label']
        self._input = config['input']
        self._weight = config.get('weight', None)
        self._metric_label = config.get('metric_label', None)
        self._bound = config.get('bound', [-15.0, 15.0])
        self._extend_output = {
            'metric_label': self._metric_label,
            'metric_dict': {
                'auc': {'var': None},
                'batch_auc': {'var': None},
                'stat_pos': {'var': None, 'data_type': 'int64'},
                'stat_neg': {'var': None, 'data_type': 'int64'},
                'batch_stat_pos': {'var': None, 'data_type': 'int64'},
                'batch_stat_neg': {'var': None, 'data_type': 'int64'},
                'pos_ins_num': {'var': None},
                'abserr': {'var': None},
                'sqrerr': {'var': None},
                'prob': {'var': None},
                'total_ins_num': {'var': None},
                'q': {'var': None}
            }
        }
        

    def generate_fluid(self, param): 
        """R
        """
        input_layer = param['layer'][self._input[0]]
        label_layer = param['layer'][self._label]
        output = fluid.layers.clip(input_layer, self._bound[0], self._bound[1], name=self._name)
        norm = fluid.layers.sigmoid(output, name=self._name)
        output = fluid.layers.log_loss(norm, fluid.layers.cast(x=label_layer, dtype='float32'))
        if self._weight:
            weight_layer = param['layer'][self._weight]
            output = fluid.layers.elementwise_mul(output, weight_layer)
        output = fluid.layers.mean(x=output)
        self._extend_output['loss'] = output
        
        #For AUC Metric
        metric = self._extend_output['metric_dict']
        binary_predict = fluid.layers.concat(
            input=[fluid.layers.elementwise_sub(fluid.layers.ceil(norm), norm), norm], axis=1)
        metric['auc']['var'], metric['batch_auc']['var'], [metric['batch_stat_pos']['var'], \
        metric['batch_stat_neg']['var'], metric['stat_pos']['var'], metric['stat_neg']['var']] = \
            fluid.layers.auc(input=binary_predict, label=fluid.layers.cast(x=label_layer, dtype='int64'), \
            curve='ROC', num_thresholds=32)

        metric['sqrerr']['var'], metric['abserr']['var'], metric['prob']['var'], metric['q']['var'], \
        metric['pos_ins_num']['var'], metric['total_ins_num']['var'] = \
            fluid.contrib.layers.ctr_metric_bundle(norm, fluid.layers.cast(x=label_layer, dtype='float32'))

        return norm, self._extend_output
