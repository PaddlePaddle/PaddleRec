
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet

class ModelJoinCommon(object):
    def __init__(self, slot_file_name,  slot_common_file_name, all_slot_file, join_ins_tag):
        self.slot_file_name = slot_file_name
        self.slot_common_file_name = slot_common_file_name
        self.dict_dim = 10 # it's fake
        self.emb_dim = 9 + 2
        self.init_range = 0.2
        self.all_slot_file = all_slot_file
        self.ins_tag_v = join_ins_tag
        self._train_program = fluid.Program()
        self._startup_program = fluid.Program()
        with fluid.program_guard(self._train_program, self._startup_program):
            with fluid.unique_name.guard():
                self.show = fluid.layers.data(name="show", shape=[-1, 1], dtype="int64", lod_level=0, append_batch_size=False)
                self.label = fluid.layers.data(name="click", shape=[-1, 1], dtype="int64", lod_level=0, append_batch_size=False)
                self.ins_weight = fluid.layers.data(
                    name="12345",
                    shape=[-1, 1],
                    dtype="float32",
                    lod_level=0,
                    append_batch_size=False,
                    stop_gradient=True)
                self.ins_tag = fluid.layers.data(
                    name="23456",
                    shape=[-1, 1],
                    dtype="int64",
                    lod_level=0,
                    append_batch_size=False,
                    stop_gradient=True)
                self.x3_ts = fluid.layers.create_global_var(shape=[1,1], value=self.ins_tag_v, dtype='int64', persistable=True, force_cpu=True, name='X3')
                self.x3_ts.stop_gradient=True
                self.label_after_filter, self.filter_loss = fluid.layers.filter_by_instag(self.label, self.ins_tag, self.x3_ts, True)
                self.label_after_filter.stop_gradient=True
                self.show_after_filter, _ = fluid.layers.filter_by_instag(self.show, self.ins_tag, self.x3_ts, True)
                self.show_after_filter.stop_gradient=True
                self.ins_weight_after_filter, _ = fluid.layers.filter_by_instag(self.ins_weight, self.ins_tag,  self.x3_ts, True)
                self.ins_weight_after_filter.stop_gradient=True
                
                self.slots_name = []
                for line in open(self.slot_file_name, 'r'):
                    slot = line.strip()
                    self.slots_name.append(slot)

                self.all_slots_name = []
                for line in open(self.all_slot_file, 'r'):
                    self.all_slots_name.append(line.strip())

                self.slots = []
                self.embs = []
                for i in self.all_slots_name:
                    if i == self.ins_weight.name or i == self.ins_tag.name:
                        pass
                    elif i not in self.slots_name:
                        pass
                    else:
                        l = fluid.layers.data(name=i, shape=[1], dtype="int64", lod_level=1)
                        emb = fluid.layers.embedding(input=l, size=[self.dict_dim, self.emb_dim], is_sparse = True, is_distributed=True, param_attr=fluid.ParamAttr(name="embedding"))
                        self.slots.append(l)
                        self.embs.append(emb)

                self.common_slot_name = []
                for i in open(self.slot_common_file_name, 'r'):
                    self.common_slot_name.append(i.strip())

                cvms = []
                cast_label = fluid.layers.cast(self.label, dtype='float32')
                cast_label.stop_gradient = True
                ones = fluid.layers.fill_constant_batch_size_like(input=self.label, shape=[-1, 1], dtype="float32", value=1)
                show_clk = fluid.layers.cast(fluid.layers.concat([ones, cast_label], axis=1), dtype='float32')
                show_clk.stop_gradient = True
                for index in range(len(self.embs)):
                    emb = self.embs[index]
                    emb.stop_gradient=True
                    bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
                    bow.stop_gradient=True
                    cvm = fluid.layers.continuous_value_model(bow, show_clk, True)
                    cvm.stop_gradient=True
                    cvms.append(cvm)
                concat_join = fluid.layers.concat(cvms, axis=1)
                concat_join.stop_gradient=True
                
                cvms_common = []
                for index in range(len(self.common_slot_name)):
                    cvms_common.append(cvms[index])
                concat_common = fluid.layers.concat(cvms_common, axis=1)
                concat_common.stop_gradient=True
                
                bn_common = fluid.layers.data_norm(input=concat_common, name="common", epsilon=1e-4, param_attr={"batch_size":1e4,"batch_sum_default":0.0,"batch_square":1e4})

                concat_join, _ = fluid.layers.filter_by_instag(concat_join, self.ins_tag, self.x3_ts, False)
                concat_join.stop_gradient=True
                bn_join = fluid.layers.data_norm(input=concat_join, name="join", epsilon=1e-4, param_attr={"batch_size":1e4,"batch_sum_default":0.0,"batch_square":1e4})
                
                join_fc = self.fcs(bn_join, "join")
                join_similarity_norm = fluid.layers.sigmoid(fluid.layers.clip(join_fc, min=-15.0, max=15.0), name="join_similarity_norm")
                join_cost = fluid.layers.log_loss(input=join_similarity_norm, label=fluid.layers.cast(x=self.label_after_filter, dtype='float32'))
                join_cost = fluid.layers.elementwise_mul(join_cost, self.ins_weight_after_filter)
                join_cost = fluid.layers.elementwise_mul(join_cost, self.filter_loss)
                join_avg_cost = fluid.layers.mean(x=join_cost)

                common_fc = self.fcs(bn_common, "common")
                common_similarity_norm = fluid.layers.sigmoid(fluid.layers.clip(common_fc, min=-15.0, max=15.0), name="common_similarity_norm")
                common_cost = fluid.layers.log_loss(input=common_similarity_norm, label=fluid.layers.cast(x=self.label, dtype='float32'))
                common_cost = fluid.layers.elementwise_mul(common_cost, self.ins_weight)
                common_avg_cost = fluid.layers.mean(x=common_cost)

                self.joint_cost = join_avg_cost + common_avg_cost

                join_binary_predict = fluid.layers.concat(
                        input=[fluid.layers.elementwise_sub(fluid.layers.ceil(join_similarity_norm), join_similarity_norm), join_similarity_norm], axis=1)
                self.join_auc, batch_auc, [self.join_batch_stat_pos, self.join_batch_stat_neg, self.join_stat_pos, self.join_stat_neg] = \
                        fluid.layers.auc(input=join_binary_predict, label=self.label_after_filter, curve='ROC', num_thresholds=4096)
                self.join_sqrerr, self.join_abserr, self.join_prob, self.join_q, self.join_pos, self.join_total = \
                        fluid.contrib.layers.ctr_metric_bundle(join_similarity_norm, fluid.layers.cast(x=self.label_after_filter, dtype='float32'))

                common_binary_predict = fluid.layers.concat(
                        input=[fluid.layers.elementwise_sub(fluid.layers.ceil(common_similarity_norm), common_similarity_norm), common_similarity_norm], axis=1)
                self.common_auc, batch_auc, [self.common_batch_stat_pos, self.common_batch_stat_neg, self.common_stat_pos, self.common_stat_neg] = \
                        fluid.layers.auc(input=common_binary_predict, label=self.label, curve='ROC', num_thresholds=4096)
                self.common_sqrerr, self.common_abserr, self.common_prob, self.common_q, self.common_pos, self.common_total = \
                        fluid.contrib.layers.ctr_metric_bundle(common_similarity_norm, fluid.layers.cast(x=self.label, dtype='float32'))

        self.tmp_train_program = fluid.Program()
        self.tmp_startup_program = fluid.Program()
        with fluid.program_guard(self.tmp_train_program, self.tmp_startup_program):
            with fluid.unique_name.guard():
                self._all_slots = [self.show, self.label]
                self._merge_slots = []
                for i in self.all_slots_name:
                    if i == self.ins_weight.name:
                        self._all_slots.append(self.ins_weight)
                    elif i == self.ins_tag.name:
                        self._all_slots.append(self.ins_tag)
                    else:
                        l = fluid.layers.data(name=i, shape=[1], dtype="int64", lod_level=1)
                        self._all_slots.append(l)
                        self._merge_slots.append(l)


    def fcs(self, bn, prefix):
        fc_layers_input = [bn]
        fc_layers_size = [511, 255, 255, 127, 127, 127, 127, 1]
        fc_layers_act = ["relu"] * (len(fc_layers_size) - 1) + [None]
        scales_tmp = [bn.shape[1]] + fc_layers_size
        scales = []
        for i in range(len(scales_tmp)):
            scales.append(self.init_range / (scales_tmp[i] ** 0.5))
        for i in range(len(fc_layers_size)):
            name = prefix+"_"+str(i)
            fc = fluid.layers.fc(
                    input = fc_layers_input[-1],
                    size = fc_layers_size[i],
                    act = fc_layers_act[i],
                    param_attr = \
                            fluid.ParamAttr(learning_rate=1.0, \
                            initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0 * scales[i])),
                    bias_attr = \
                            fluid.ParamAttr(learning_rate=1.0, \
                            initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0 * scales[i])),
                    name=name)
            fc_layers_input.append(fc)
        return fc_layers_input[-1]
