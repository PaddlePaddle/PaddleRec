
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet

class Model(object):
    def __init__(self, slot_file_name, all_slot_file, use_cvm, ins_tag, is_update_model):
        self._slot_file_name = slot_file_name
        self._use_cvm = use_cvm
        self._dict_dim = 10 # it's fake
        self._emb_dim = 9 + 2
        self._init_range = 0.2
        self._all_slot_file = all_slot_file
        self._not_use_slots = []
        self._not_use_slotemb = []
        self._all_slots = []
        self._ins_tag_value = ins_tag
        self._is_update_model = is_update_model
        self._train_program = fluid.Program()
        self._startup_program = fluid.Program()
        self.save_vars = []
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
                self.slots = []
                self.slots_name = []
                self.embs = []

                
                if self._ins_tag_value != 0:
                    self.x3_ts = fluid.layers.create_global_var(shape=[1,1], value=self._ins_tag_value, dtype='int64', persistable=True, force_cpu=True, name='X3')
                    self.x3_ts.stop_gradient=True
                    self.label_after_filter, self.filter_loss = fluid.layers.filter_by_instag(self.label, self.ins_tag, self.x3_ts, True)
                    self.label_after_filter.stop_gradient=True
                    self.show_after_filter, _ = fluid.layers.filter_by_instag(self.show, self.ins_tag, self.x3_ts, True)
                    self.show_after_filter.stop_gradient=True
                    self.ins_weight_after_filter, _ = fluid.layers.filter_by_instag(self.ins_weight, self.ins_tag,  self.x3_ts, True)
                    self.ins_weight_after_filter.stop_gradient=True

                for line in open(self._slot_file_name, 'r'):
                    slot = line.strip()
                    self.slots_name.append(slot)

                self.all_slots_name = []
                for line in open(self._all_slot_file, 'r'):
                    self.all_slots_name.append(line.strip())
                for i in self.all_slots_name:
                    if i == self.ins_weight.name or i == self.ins_tag.name:
                        pass
                    elif i not in self.slots_name:
                        pass
                    else:
                        l = fluid.layers.data(name=i, shape=[1], dtype="int64", lod_level=1)
                        emb = fluid.layers.embedding(input=l, size=[self._dict_dim, self._emb_dim], is_sparse = True, is_distributed=True, param_attr=fluid.ParamAttr(name="embedding"))
                        self.slots.append(l)
                        self.embs.append(emb)

                if self._ins_tag_value != 0:
                    self.emb = self.slot_net(self.slots, self.label_after_filter)
                else:
                    self.emb = self.slot_net(self.slots, self.label)

                self.similarity_norm = fluid.layers.sigmoid(fluid.layers.clip(self.emb, min=-15.0, max=15.0), name="similarity_norm")
                
                if self._ins_tag_value != 0:
                    self.cost = fluid.layers.log_loss(input=self.similarity_norm, label=fluid.layers.cast(x=self.label_after_filter, dtype='float32'))
                else:
                    self.cost = fluid.layers.log_loss(input=self.similarity_norm, label=fluid.layers.cast(x=self.label, dtype='float32'))
               
                if self._ins_tag_value != 0:
                    self.cost = fluid.layers.elementwise_mul(self.cost, self.ins_weight_after_filter)
                else:
                    self.cost = fluid.layers.elementwise_mul(self.cost, self.ins_weight)
                
                if self._ins_tag_value != 0:
                    self.cost = fluid.layers.elementwise_mul(self.cost, self.filter_loss)

                self.avg_cost = fluid.layers.mean(x=self.cost)

                binary_predict = fluid.layers.concat(
                        input=[fluid.layers.elementwise_sub(fluid.layers.ceil(self.similarity_norm), self.similarity_norm), self.similarity_norm], axis=1)
                
                if self._ins_tag_value != 0:
                    self.auc, batch_auc, [self.batch_stat_pos, self.batch_stat_neg, self.stat_pos, self.stat_neg] = \
                            fluid.layers.auc(input=binary_predict, label=self.label_after_filter, curve='ROC', num_thresholds=4096)
                    self.sqrerr, self.abserr, self.prob, self.q, self.pos, self.total = \
                    fluid.contrib.layers.ctr_metric_bundle(self.similarity_norm, fluid.layers.cast(x=self.label_after_filter, dtype='float32'))

                    #self.precise_ins_num = fluid.layers.create_global_var(persistable=True, dtype='float32', shape=[1])
                    #batch_ins_num = fluid.layers.reduce_sum(self.filter_loss)
                    #self.precise_ins_num = fluid.layers.elementwise_add(batch_ins_num, self.precise_ins_num)

                else:
                    self.auc, batch_auc, [self.batch_stat_pos, self.batch_stat_neg, self.stat_pos, self.stat_neg] = \
                            fluid.layers.auc(input=binary_predict, label=self.label, curve='ROC', num_thresholds=4096)
                    self.sqrerr, self.abserr, self.prob, self.q, self.pos, self.total = \
                    fluid.contrib.layers.ctr_metric_bundle(self.similarity_norm, fluid.layers.cast(x=self.label, dtype='float32'))



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




    def slot_net(self, slots, label, lr_x=1.0):
        input_data = []
        cvms = []

        cast_label = fluid.layers.cast(label, dtype='float32')
        cast_label.stop_gradient = True
        ones = fluid.layers.fill_constant_batch_size_like(input=label, shape=[-1, 1], dtype="float32", value=1)
        show_clk = fluid.layers.cast(fluid.layers.concat([ones, cast_label], axis=1), dtype='float32')
        show_clk.stop_gradient = True

        for index in range(len(slots)):
            input_data.append(slots[index])
            emb = self.embs[index]
            bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
            cvm = fluid.layers.continuous_value_model(bow, show_clk, self._use_cvm)
            cvms.append(cvm)

        concat = None
        if self._ins_tag_value != 0:
            concat = fluid.layers.concat(cvms, axis=1)
            concat, _ = fluid.layers.filter_by_instag(concat, self.ins_tag, self.x3_ts, False)
        else:
            concat = fluid.layers.concat(cvms, axis=1)
        bn = concat
        if self._use_cvm:
            bn = fluid.layers.data_norm(input=concat, name="bn6048", epsilon=1e-4,
                    param_attr={
                        "batch_size":1e4,
                        "batch_sum_default":0.0,
                        "batch_square":1e4})
            self.save_vars.append(bn)
        fc_layers_input = [bn]
        if self._is_update_model:
            fc_layers_size = [511, 255, 127, 127, 127, 1]
        else:
            fc_layers_size = [511, 255, 255, 127, 127, 127, 127, 1]
        fc_layers_act = ["relu"] * (len(fc_layers_size) - 1) + [None]
        scales_tmp = [bn.shape[1]] + fc_layers_size
        scales = []
        for i in range(len(scales_tmp)):
            scales.append(self._init_range / (scales_tmp[i] ** 0.5))
        for i in range(len(fc_layers_size)):
            fc = fluid.layers.fc(
                    input = fc_layers_input[-1],
                    size = fc_layers_size[i],
                    act = fc_layers_act[i],
                    param_attr = \
                            fluid.ParamAttr(learning_rate=lr_x, \
                            initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0 * scales[i])),
                    bias_attr = \
                            fluid.ParamAttr(learning_rate=lr_x, \
                            initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0 * scales[i])))
            fc_layers_input.append(fc)
            self.save_vars.append(fc)
        return fc_layers_input[-1]
