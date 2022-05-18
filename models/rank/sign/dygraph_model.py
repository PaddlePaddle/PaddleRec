# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import pgl
import os
import numpy as np

from net import L0_SIGN


def get_n_feature(config):
    data_dir = config.get("runner.train_data_dir", "data")
    if os.path.split(os.getcwd())[-1] != 'sign':
        data_dir = os.path.join(os.getcwd(), "models/rank/sign", data_dir)
    file_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    max_node_index = 0
    for file in file_list:
        with open(file, 'r') as f:
            for line in f:
                data = line.split()
                # all other nums is node
                node_list = [int(data[i]) for i in range(len(data))[1:]]
                max_node_index = max(max_node_index, max(node_list))
    return max_node_index + 1


class DygraphModel():
    # define model
    def create_model(self, config):
        pred_edges = config.get('hyper_parameters.pred_edges', 1)
        dim = config.get('hyper_parameters.dim', 8)
        hidden_layer = config.get('hyper_parameters.hidden_layer', 64)
        l0_para = config.get('hyper_parameters.l0_para', [0.66, -0.1, 1.1])
        batch_size = config.get('runner.train_batch_size', 8)
        n_feature = get_n_feature(config=config)

        model = L0_SIGN(pred_edges, n_feature, dim, hidden_layer, l0_para,
                        batch_size)

        return model

    # define feeds which convert numpy of batch data to paddle.tensor 
    def create_feeds(self, batch_data, config):
        batch_size = config.get("runner.train_batch_size", 1024)
        graphs = []
        labels = []
        for i in range(batch_size):
            g = pgl.Graph(
                num_nodes=batch_data[0][i].numpy(),
                edges=batch_data[1][i].numpy(),
                node_feat={"node_attr": batch_data[2][i].numpy()},
                edge_feat={"edge_attr": batch_data[3][i].numpy()})
            graphs.append(g)
            labels.append(batch_data[4][i].numpy())
        graphs = pgl.Graph.batch(graphs).tensor()
        labels = paddle.to_tensor(labels, dtype='float32')
        edges = np.array(graphs.edges, dtype="int32")
        node_feat = np.array(graphs.node_feat["node_attr"], dtype="int32")
        edge_feat = np.array(graphs.edge_feat["edge_attr"], dtype="int32")
        segment_ids = graphs.graph_node_id

        return edges, node_feat, edge_feat, segment_ids, labels

    # define loss function by predicts and label
    def create_loss(self, output, label, l0_penaty, l2_penaty, l0_weight,
                    l2_weight):
        crit = paddle.nn.MSELoss()
        baseloss = crit(output, label)
        l0_loss = l0_penaty * l0_weight
        l2_loss = l2_penaty * l2_weight
        loss = baseloss + l0_loss + l2_loss
        # loss_all += num_graph * loss.item()

        return loss

    # define optimizer 
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.05)
        optimizer = paddle.optimizer.Adagrad(
            learning_rate=lr,
            parameters=dy_model.parameters(),
            epsilon=1e-05,
            weight_decay=1e-05)
        return optimizer

    # define metrics such as auc/acc
    def create_metrics(self):
        metrics_list_name = ["AUC", "ACC"]
        auc_metric = paddle.metric.Auc()
        acc_metric = paddle.metric.Accuracy()
        metrics_list = [auc_metric, acc_metric]
        return metrics_list, metrics_list_name

    # construct train forward phase  
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        edges, node_feat, edge_feat, segment_ids, labels = self.create_feeds(
            batch_data, config)
        # predict
        output, l0_penaty, l2_penaty = dy_model.forward(
            edges, node_feat, edge_feat, segment_ids, True)
        # get loss
        l0_weight = config.get("hyper_parameters.l0_weight", 0.001)
        l2_weight = config.get("hyper_parameters.l0_weight", 0.001)
        loss = self.create_loss(output, labels, l0_penaty, l2_penaty,
                                l0_weight, l2_weight)
        # update metrics
        predictions = np.vstack(output)
        labels = np.vstack(labels)
        labels = labels[:, 1].reshape((-1, 1))
        metrics_list[0].update(preds=predictions, labels=labels)
        correct = metrics_list[1].compute(
            paddle.to_tensor(predictions), paddle.to_tensor(labels))
        metrics_list[1].update(correct)
        # print dict
        print_dict = {'loss': loss}
        return loss, metrics_list, print_dict

    # construct infer forward phase  
    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        edges, node_feat, edge_feat, segment_ids, labels = self.create_feeds(
            batch_data, config)
        # predict
        output, _, _ = dy_model.forward(edges, node_feat, edge_feat,
                                        segment_ids, False)
        # update metrics
        predictions = np.vstack(output)
        labels = np.vstack(labels)
        labels = labels[:, 1].reshape((-1, 1))
        metrics_list[0].update(preds=predictions, labels=labels)
        correct = metrics_list[1].compute(
            paddle.to_tensor(predictions), paddle.to_tensor(labels))
        metrics_list[1].update(correct)
        return metrics_list, None
