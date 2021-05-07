from __future__ import print_function
import numpy as np

from paddle.io import IterableDataset
import random

class RecDataset(IterableDataset):
    def __init__(self, file_list, config, graph_index, mode = "train"):
        super(RecDataset, self).__init__()
        self.file_list = file_list
        self.init()
        self.use_multi_task_learning = config.get("hyper_parameters.use_multi_task_learning")
        self.item_count = config.get("hyper_parameters.item_count")
        self.mode = mode
        self.graph_index = graph_index
        self.config = config


    def init(self):
        pass

    def __iter__(self):
        for file in self.file_list:
            with open(file, "r") as rf:
                for l in rf:
                    line = l.strip().split()
                    item_id = int(line[-1])
                    user_embedding = []
                    for i in line[:-1]:
                        user_embedding.append(float(i))
                    output_list = []
                    output_list.append(np.array([item_id]).astype("int32"))
                    output_list.append(
                        np.array(user_embedding).astype('float32'))
                    path_set = self.graph_index.get_path_of_item(item_id)
                    #print("path_set -->",path_set)
                    item_path_kd_label = []
                    for path in path_set[0]:
                            path_label = np.array(self.graph_index.path_id_to_kd_represent(path))
                            item_path_kd_label.append(path_label)
                    output_list.append(np.array(item_path_kd_label).astype("int32"))                
                    if self.use_multi_task_learning:
                        label = [item_id]
                        output_list.append(
                            np.array(label).astype('int32'))                           
                        output_list.append(np.array([(item_id + self.item_count/2 ) % self.item_count]).astype("int32"))
                    yield output_list