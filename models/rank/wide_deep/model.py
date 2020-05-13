import paddle.fluid as fluid
import math

from paddlerec.core.utils import envs
from paddlerec.core.model import Model as ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)
    
    def wide_part(self, data):
        out = fluid.layers.fc(input=data,
                            size=1, 
                            param_attr=fluid.ParamAttr(initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=1.0 / math.sqrt(data.shape[1])),
                                                       regularizer=fluid.regularizer.L2DecayRegularizer(regularization_coeff=1e-4)),
                            act=None,
                            name='wide')
        return out
        
    def fc(self, data, hidden_units, active, tag):
        output = fluid.layers.fc(input=data,
                            size=hidden_units, 
                            param_attr=fluid.ParamAttr(initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=1.0 / math.sqrt(data.shape[1]))),
                            act=active,
                            name=tag)
                            
        return output
        
    def deep_part(self, data, hidden1_units, hidden2_units, hidden3_units):
        l1 = self.fc(data, hidden1_units, 'relu', 'l1')
        l2 = self.fc(l1, hidden2_units, 'relu', 'l2')
        l3 = self.fc(l2, hidden3_units, 'relu', 'l3')
        
        return l3
    
    def train_net(self):
        wide_input = fluid.data(name='wide_input', shape=[None, 8], dtype='float32')
        deep_input = fluid.data(name='deep_input', shape=[None, 58], dtype='float32')
        label = fluid.data(name='label', shape=[None, 1], dtype='float32')
        self._data_var.append(wide_input)
        self._data_var.append(deep_input)
        self._data_var.append(label)

        hidden1_units = envs.get_global_env("hyper_parameters.hidden1_units", 75, self._namespace)
        hidden2_units = envs.get_global_env("hyper_parameters.hidden2_units", 50, self._namespace)
        hidden3_units = envs.get_global_env("hyper_parameters.hidden3_units", 25, self._namespace)
        wide_output = self.wide_part(wide_input)
        deep_output = self.deep_part(deep_input, hidden1_units, hidden2_units, hidden3_units)
        
        wide_model = fluid.layers.fc(input=wide_output,
                        size=1, 
                        param_attr=fluid.ParamAttr(initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=1.0)),
                        act=None,
                        name='w_wide')
                        
        deep_model = fluid.layers.fc(input=deep_output,
                        size=1, 
                        param_attr=fluid.ParamAttr(initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=1.0)),
                        act=None,
                        name='w_deep')
                    
        prediction = fluid.layers.elementwise_add(wide_model, deep_model)
        pred = fluid.layers.sigmoid(fluid.layers.clip(prediction, min=-15.0, max=15.0), name="prediction")

        num_seqs = fluid.layers.create_tensor(dtype='int64')
        acc = fluid.layers.accuracy(input=pred, label=fluid.layers.cast(x=label, dtype='int64'), total=num_seqs)
        auc_var, batch_auc, auc_states = fluid.layers.auc(input=pred, label=fluid.layers.cast(x=label, dtype='int64'))
        
        self._metrics["AUC"] = auc_var
        self._metrics["BATCH_AUC"] = batch_auc
        self._metrics["ACC"] = acc

        cost = fluid.layers.sigmoid_cross_entropy_with_logits(x=prediction, label=label) 
        avg_cost = fluid.layers.mean(cost)
        self._cost = avg_cost

    def optimizer(self):
        learning_rate = envs.get_global_env("hyper_parameters.learning_rate", None, self._namespace)
        optimizer = fluid.optimizer.Adam(learning_rate, lazy_mode=True)
        return optimizer

    def infer_net(self, parameter_list):
        self.deepfm_net()