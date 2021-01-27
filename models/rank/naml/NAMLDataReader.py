from __future__ import print_function
import numpy as np
import re,random
from paddle.io import IterableDataset


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list
        self.browse_file_list = []
        self.article_file_list = []
        for x in file_list:
            if re.match('[\\S]*browse[0-9]*.txt$', x) != None:
                self.browse_file_list.append(x)
            elif re.match('[\\S]*article[0-9]*.txt$', x) != None:
                self.article_file_list.append(x)
        self.config = config
        self.article_content_size = config.get("hyper_parameters.article_content_size")
        self.article_title_size = config.get("hyper_parameters.article_title_size")
        self.browse_size = config.get("hyper_parameters.browse_size")
        self.neg_condidate_sample_size = config.get("hyper_parameters.neg_condidate_sample_size")
        self.word_dict_size = int(config.get("hyper_parameters.word_dict_size"))
        self.category_size = int(config.get("hyper_parameters.category_size"))
        self.sub_category_size = int(config.get("hyper_parameters.sub_category_size"))
        self.article_map_cate = {}
        self.article_map_title = {}
        self.article_map_content = {}
        self.article_map_sub_cate = {}
        self.init()

    def convert_unk(self,id):
        if id in self.article_map_cate:
            return id
        return "padding"
    def init(self):
        self.article_map_cate["padding"] = self.category_size
        self.article_map_sub_cate["padding"] = self.sub_category_size
        self.article_map_title["padding"] = [self.word_dict_size] * self.article_title_size
        self.article_map_content["padding"] = [self.word_dict_size]* self.article_content_size
        #line [0]id cate_id sub_cate_id [3]title content
        for file in self.article_file_list:
            with open(file,"r") as rf:
                for l in rf:
                    line = l.strip().split('\t')
                    id = line[0]
                    #line 0 cate   1:subcate,  2:title, 3 content;
                    line = [[int(line[1])],[int(line[2])],[int(t) for t in line[3].split(" ")],[int(t) for t in line[4].split(" ")]]
                    line[2] += [self.word_dict_size] * (self.article_title_size - len(line[2]))
                    line[3] += [self.word_dict_size] * (self.article_content_size - len(line[3]))
                    self.article_map_cate[id] = line[0][0]
                    self.article_map_sub_cate[id] = line[1][0]
                    if len(line[2]) > self.article_title_size:
                        line[2] = line[2][:self.article_title_size]
                    if len(line[3]) > self.article_content_size:
                        line[3] = line[3][:self.article_content_size]
                    self.article_map_title[id] = line[2]
                    self.article_map_content[id] = line[3]
                    #print(id)
                #cateId,subCateId    title    content

    def __iter__(self):
        self.data = []
        for file in self.browse_file_list:
            with open(file, "r") as rf:
                for l in rf:
                    # sparse
                    line_x = l.strip().split("\t")
                    line = []
                    for i in range(3):
                        line.append(line_x[i].split(" "))
                    #line = [[line[0].split(" ")],[line[1].split(" ")],[line[2].split(" ")]]
                    # hold_out = line[0][-1]
                    # line[0][-1] = 0
                    if len(line[0]) > self.browse_size:
                        line[0] = line[0][len(line[0]) - self.browse_size:]
                    line[0] += ["unk"] * (self.browse_size - len(line[0]))
                    neg_candidate = line[2]
                    if len(neg_candidate) < self.neg_condidate_sample_size:
                        continue;
                    candidate = neg_candidate[:self.neg_condidate_sample_size]
                    candidate.append(line[1][0])
                    line[1] = []
                    ids = list(range(self.neg_condidate_sample_size + 1))
                    random.shuffle(ids)
                    label = []
                    for i in ids:
                        line[1].append(candidate[i]) #1 condidate 0:visit
                        if i == self.neg_condidate_sample_size:
                            label.append(1)
                        else:
                            label.append(0)

                    article_list = [np.array(label)]
#                    l = [self.article_map[i] for i in line[1]]
                    article_list.append(np.array([self.article_map_cate[self.convert_unk(i)] for i in line[1]]))
                    article_list.append(np.array([self.article_map_cate[self.convert_unk(i)] for i in line[0]]))
                    article_list.append(np.array([self.article_map_sub_cate[self.convert_unk(i)] for i in line[1]]))
                    article_list.append(np.array([self.article_map_sub_cate[self.convert_unk(i)] for i in line[0]]))
                    article_list.append(np.array([self.article_map_title[self.convert_unk(i)] for i in line[1]]))
                    article_list.append(np.array([self.article_map_title[self.convert_unk(i)] for i in line[0]]))
                    article_list.append(np.array([self.article_map_content[self.convert_unk(i)] for i in line[1]]))
                    article_list.append(np.array([self.article_map_content[self.convert_unk(i)] for i in line[0]]))
                    #output_list = [article_list,None]
                    yield article_list
