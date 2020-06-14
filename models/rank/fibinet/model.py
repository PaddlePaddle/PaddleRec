import paddle.fluid as fluid
import itertools
from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase

class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)
    def _init_hyper_parameters(self):
        self.is_distributed = True if envs.get_fleet_mode().upper() == "PSLIB" else False
        self.sparse_feature_number = envs.get_global_env(
            "hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = envs.get_global_env(
            "hyper_parameters.sparse_feature_dim")
        self.learning_rate = envs.get_global_env(
            "hyper_parameters.optimizer.learning_rate")

    def fc(self, data, size, active, tag):
        output = fluid.layers.fc(input=data,
                            size=size, 
                            param_attr=fluid.initializer.Xavier(uniform=False),
                            act=active,
                            name=tag)
                            
        return output
    def SENETLayer(self, inputs, filed_size, reduction_ratio = 3):
        reduction_size = max(1, filed_size // reduction_ratio)
        Z = fluid.layers.reduce_mean(inputs, dim=-1)
        
        A_1 = self.fc(Z, reduction_size, 'relu', 'W_1')
        A_2 = self.fc(A_1, filed_size, 'relu', 'W_2')

        V = fluid.layers.elementwise_mul(inputs, y = fluid.layers.unsqueeze(input=A_2, axes=[2]))
        
        return fluid.layers.split(V, num_or_sections=filed_size, dim=1)

    def BilinearInteraction(self, inputs, filed_size, embedding_size, bilinear_type="interaction"):
        if bilinear_type == "all":
            p = [fluid.layers.elementwise_mul(self.fc(v_i, embedding_size, None, None), fluid.layers.squeeze(input=v_j, axes=[1])) for v_i, v_j in itertools.combinations(inputs, 2)]
        else:
            raise NotImplementedError

        return fluid.layers.concat(input=p, axis=1)

    def DNNLayer(self, inputs, dropout_rate=0.5):
        deep_input = inputs
        for i, hidden_unit in enumerate([400, 400, 400]):
            fc_out = self.fc(deep_input, hidden_unit, 'relu', 'd_' + str(i))
            fc_out = fluid.layers.dropout(fc_out, dropout_prob=dropout_rate)
            deep_input = fc_out

        return deep_input

    def net(self, input, is_infer=False): 
        self.sparse_inputs = self._sparse_data_var[1:]
        self.dense_input = self._dense_data_var[0]
        self.label_input = self._sparse_data_var[0]

        def embedding_layer(input):
            emb = fluid.layers.embedding(
                input=input,
                is_sparse=True,
                is_distributed=self.is_distributed,
                size=[self.sparse_feature_number, self.sparse_feature_dim],
                param_attr=fluid.ParamAttr(
                    name="SparseFeatFactors",
                    initializer=fluid.initializer.Uniform()), )
            emb_sum = fluid.layers.sequence_pool(input=emb, pool_type='sum')
            return emb_sum
        concat_emb = list(map(embedding_layer, self.sparse_inputs))

        filed_size = len(self.sparse_inputs)
        bilinear_type = envs.get_global_env("hyper_parameters.bilinear_type")
        reduction_ratio = envs.get_global_env("hyper_parameters.reduction_ratio")
        dropout_rate = envs.get_global_env("hyper_parameters.dropout_rate")

        senet_output = self.SENETLayer(concat_emb, filed_size, reduction_ratio)
        senet_bilinear_out = self.BilinearInteraction(senet_output, filed_size, self.sparse_feature_dim, bilinear_type)

        concat_emb = fluid.layers.split(concat_emb, num_or_sections=filed_size, dim=1)
        bilinear_out = self.BilinearInteraction(concat_emb, filed_size, self.sparse_feature_dim, bilinear_type)
        dnn_input = fluid.layers.concat(input=[senet_bilinear_out, bilinear_out, self.dense_input], axis=1)
        dnn_output = self.DNNLayer(dnn_input, dropout_rate)
 
        y_pred = self.fc(dnn_output, 1, 'sigmoid', 'logit')
        self.predict = y_pred
        auc, batch_auc, _ = fluid.layers.auc(input=self.predict,
                                             label=self.label_input,
                                             num_thresholds=2**12,
                                             slide_steps=20)

        if is_infer:
            self._infer_results["AUC"] = auc
            self._infer_results["BATCH_AUC"] = batch_auc
            return

        self._metrics["AUC"] = auc
        self._metrics["BATCH_AUC"] = batch_auc
        cost = fluid.layers.log_loss(input=self.predict, label=fluid.layers.cast(x=self.label_input, dtype='float32'))
        avg_cost = fluid.layers.reduce_mean(cost)
        self._cost = avg_cost

        def optimizer(self):
            optimizer = fluid.optimizer.Adam(self.learning_rate, lazy_mode=True)
            return optimizer

        def infer_net(self):
            pass
