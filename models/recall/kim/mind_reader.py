from hypers import *
from datetime import datetime
import time
from nltk.tokenize import word_tokenize
import numpy as np
import os
import json
import random
from paddle.io import Dataset


class RecDataset(Dataset):
    def __init__(self, file_list, config):
        KG_root_path = embedding_path = data_root_path = config['runner.train_data_dir']
        news, news_index, category_dict, subcategory_dict, word_dict = read_news(data_root_path, 'docs.tsv')
        print(len(word_dict))
        news_title, news_vert, news_subvert = get_doc_input(news, news_index, category_dict, subcategory_dict,
                                                            word_dict)
        graph, EntityId2Index, EntityIndex2Id, entity_embedding = load_entity_metadata(KG_root_path)
        news_entity = load_news_entity(news, EntityId2Index, data_root_path)
        news_entity_index = parse_zero_hop_entity(EntityId2Index, news_entity, news_index, max_entity_num)
        one_hop_entity = parse_one_hop_entity(EntityId2Index, EntityIndex2Id, news_entity_index, graph, news_index,
                                              max_entity_num)
        mode = config.get('mode', 'train')
        if mode == 'train':
            train_session = read_clickhistory(data_root_path, news_index, 'train.tsv')
            train_user = parse_user(news_index, train_session)
            train_sess, train_user_id, train_label = get_train_input(news_index, train_session)
            self.dataset = TrainDataset(news_title, news_entity_index, one_hop_entity, entity_embedding,
                                        train_user['click'], train_user_id, train_sess, train_label, 16)
            title_word_embedding_matrix, have_word = load_matrix(embedding_path, word_dict)
            self.title_word_embedding_matrix = title_word_embedding_matrix.astype('float32')

        elif mode == 'test':
            test_session = read_clickhistory(data_root_path, news_index, 'test.tsv')
            test_user = parse_user(news_index, test_session)
            test_docids, test_userids, test_labels, test_bound = get_test_input(news_index, test_session)
            self.dataset = TestDataset(test_docids, test_userids, news_title, news_entity_index, one_hop_entity,
                                       entity_embedding, test_user['click'], 64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


class TrainDataset(Dataset):
    def __init__(self, news_title,news_entity_index,one_hop_entity,entity_embedding, clicked_news,user_id, news_id, label, batch_size):
        self.title = news_title

        self.clicked_news = clicked_news
        self.news_entity_index = news_entity_index
        self.one_hop_entity = one_hop_entity
        self.entity_embedding = entity_embedding

        self.user_id = user_id
        self.doc_id = news_id
        self.label = label

        self.batch_size = batch_size
        self.ImpNum = self.label.shape[0]

    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    def __get_news(self, docids):
        title = self.title[docids]
        entity_ids = self.news_entity_index[docids]
        entity_embedding = self.entity_embedding[entity_ids]

        one_hop_ids = self.one_hop_entity[docids]
        one_hop_embedding = self.entity_embedding[one_hop_ids]

        return title, entity_embedding, one_hop_embedding

    def __getitem__(self, idx):
        start = idx * self.batch_size
        ed = (idx + 1) * self.batch_size
        if ed > self.ImpNum:
            ed = self.ImpNum

        doc_ids = self.doc_id[start:ed]
        title, entity_embedding, one_hop_embedding = self.__get_news(doc_ids)

        user_ids = self.user_id[start:ed]
        clicked_ids = self.clicked_news[user_ids]

        user_title, user_entity_embedding, user_one_hop = self.__get_news(clicked_ids)

        label = self.label[start:ed]

        return title, entity_embedding, one_hop_embedding, user_title, user_entity_embedding, user_one_hop, label


class TestDataset(Dataset):
    def __init__(self, docids, userids, news_title, news_entity_index, one_hop_entity, entity_embedding, clicked_news,
                 batch_size):
        self.docids = docids
        self.userids = userids

        self.title = news_title
        self.clicked_news = clicked_news
        self.news_entity_index = news_entity_index
        self.one_hop_entity = one_hop_entity

        self.entity_embedding = entity_embedding

        self.batch_size = batch_size
        self.ImpNum = self.docids.shape[0]

    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    def __get_news(self, docids):
        title = self.title[docids]
        entity_ids = self.news_entity_index[docids]
        entity_embedding = self.entity_embedding[entity_ids]

        one_hop_ids = self.one_hop_entity[docids]
        one_hop_embedding = self.entity_embedding[one_hop_ids]

        return title, entity_embedding, one_hop_embedding

    def __getitem__(self, idx):
        start = idx * self.batch_size
        ed = (idx + 1) * self.batch_size
        if ed > self.ImpNum:
            ed = self.ImpNum

        docids = self.docids[start:ed]

        userisd = self.userids[start:ed]
        clicked_ids = self.clicked_news[userisd]

        title, entity_embedding, one_hop_embedding = self.__get_news(docids)
        user_title, user_entity_embedding, user_one_hop = self.__get_news(clicked_ids)

        return [title, entity_embedding, one_hop_embedding, user_title, user_entity_embedding, user_one_hop]


def trans2tsp(timestr):
    return int(time.mktime(datetime.strptime(timestr, '%m/%d/%Y %I:%M:%S %p').timetuple()))


def newsample(nnn, ratio):
    if ratio > len(nnn):
        return random.sample(nnn * (ratio // len(nnn) + 1), ratio)
    else:
        return random.sample(nnn, ratio)


def shuffle(pn, labeler, pos):
    index = np.arange(pn.shape[0])
    pn = pn[index]
    labeler = labeler[index]
    pos = pos[index]

    for i in range(pn.shape[0]):
        index = np.arange(npratio + 1)
        pn[i, :] = pn[i, index]
        labeler[i, :] = labeler[i, index]
    return pn, labeler, pos


def read_news(path, filenames):
    news = {}
    category = []
    subcategory = []
    news_index = {}
    index = 1
    word_dict = {}
    word_index = 1
    with open(os.path.join(path, filenames)) as f:
        lines = f.readlines()
    for line in lines:
        splited = line.strip('\n').split('\t')
        doc_id, vert, subvert, title = splited[0:4]
        news_index[doc_id] = index
        index += 1
        category.append(vert)
        subcategory.append(subvert)
        title = title.lower()
        title = word_tokenize(title)
        news[doc_id] = [vert, subvert, title]
        for word in title:
            word = word.lower()
            if not (word in word_dict):
                word_dict[word] = word_index
                word_index += 1
    category = list(set(category))
    subcategory = list(set(subcategory))
    category_dict = {}
    index = 0
    for c in category:
        category_dict[c] = index
        index += 1
    subcategory_dict = {}
    index = 0
    for c in subcategory:
        subcategory_dict[c] = index
        index += 1
    return news, news_index, category_dict, subcategory_dict, word_dict


def get_doc_input(news, news_index, category, subcategory, word_dict):
    news_num = len(news) + 1
    news_title = np.zeros((news_num, MAX_SENTENCE), dtype='int32')
    news_vert = np.zeros((news_num,), dtype='int32')
    news_subvert = np.zeros((news_num,), dtype='int32')
    for key in news:
        vert, subvert, title = news[key]
        doc_index = news_index[key]
        news_vert[doc_index] = category[vert]
        news_subvert[doc_index] = subcategory[subvert]
        for word_id in range(min(MAX_SENTENCE, len(title))):
            news_title[doc_index, word_id] = word_dict[title[word_id].lower()]

    return news_title, news_vert, news_subvert


def load_entity_metadata(KG_root_path):
    # Entity Table
    with open(os.path.join(KG_root_path, 'entity2id.txt')) as f:
        lines = f.readlines()

    EntityId2Index = {}
    EntityIndex2Id = {}
    for i in range(1, len(lines)):
        eid, eindex = lines[i].strip('\n').split('\t')
        EntityId2Index[eid] = int(eindex)
        EntityIndex2Id[int(eindex)] = eid

    entity_embedding = np.load(os.path.join(KG_root_path, 'entity_embedding.npy'))
    entity_embedding = np.concatenate([entity_embedding, np.zeros((1, 100))], axis=0)

    with open(os.path.join(KG_root_path, 'KGGraph.json')) as f:
        s = f.read()
    graph = json.loads(s)

    return graph, EntityId2Index, EntityIndex2Id, entity_embedding


def load_news_entity(news, EntityId2Index, data_root_path):
    with open(os.path.join(data_root_path, 'docs.tsv')) as f:
        lines = f.readlines()

    news_entity = {}
    g = []
    for i in range(len(lines)):
        docid, _, _, _, _, _, entities, _ = lines[i].strip('\n').split('\t')
        entities = json.loads(entities)
        news_entity[docid] = []
        for j in range(len(entities)):
            e = entities[j]['Label']
            eid = entities[j]['WikidataId']
            if not eid in EntityId2Index:
                continue
            news_entity[docid].append([e, eid, EntityId2Index[eid]])

    return news_entity


def parse_zero_hop_entity(EntityId2Index, news_entity, news_index, max_entity_num=5):
    news_entity_index = np.zeros((len(news_index) + 1, max_entity_num), dtype='int32') + len(EntityId2Index)
    for newsid in news_index:
        index = news_index[newsid]
        entities = news_entity[newsid]
        ri = np.random.permutation(len(entities))
        for j in range(min(len(entities), max_entity_num)):
            e = entities[ri[j]][-1]
            news_entity_index[index, j] = e
    return news_entity_index


def parse_one_hop_entity(EntityId2Index, EntityIndex2Id, news_entity_index, graph, news_index, max_entity_num=5):
    one_hop_entity = np.zeros((len(news_index) + 1, max_entity_num, max_entity_num), dtype='int32') + len(
        EntityId2Index)
    for newsid in news_index:
        index = news_index[newsid]
        entities = news_entity_index[index]
        for j in range(max_entity_num):
            eindex = news_entity_index[index, j]
            if eindex == len(EntityId2Index):
                continue
            eid = EntityIndex2Id[eindex]
            neighbors = graph[eid]
            rindex = np.random.permutation(len(neighbors))
            for k in range(min(max_entity_num, len(neighbors))):
                nindex = rindex[k]
                neig_id = neighbors[nindex]
                # print(neig_id)
                neig_index = EntityId2Index[neig_id]
                one_hop_entity[index, j, k] = neig_index
    return one_hop_entity


def load_matrix(embedding_path, word_dict):
    embedding_matrix = np.zeros((len(word_dict) + 1, 300))
    have_word = []
    with open(os.path.join(embedding_path, 'glove.840B.300d.txt'), 'rb') as f:
        while True:
            l = f.readline()
            if len(l) == 0:
                break
            l = l.split()
            word = l[0].decode()
            if word in word_dict:
                index = word_dict[word]
                tp = [float(x) for x in l[1:]]
                embedding_matrix[index] = np.array(tp)
                have_word.append(word)
    return embedding_matrix, have_word


def read_clickhistory(data_root_path, news_index, filename):
    lines = []
    userids = []
    with open(os.path.join(data_root_path, filename)) as f:
        lines = f.readlines()

    sessions = []
    for i in range(len(lines)):
        _, uid, eventime, click, imps = lines[i].strip().split('\t')
        if click == '':
            clikcs = []
        else:
            clikcs = click.split()
        true_click = []
        for click in clikcs:
            if not click in news_index:
                continue
            true_click.append(click)
        pos = []
        neg = []
        for imp in imps.split():
            docid, label = imp.split('-')
            if label == '1':
                pos.append(docid)
            else:
                neg.append(docid)
        sessions.append([true_click, pos, neg])
    return sessions


def parse_user(news_index, session):
    user_num = len(session)
    user = {'click': np.zeros((user_num, MAX_ALL), dtype='int32'), }
    for user_id in range(len(session)):
        tclick = []
        click, pos, neg = session[user_id]
        for i in range(len(click)):
            tclick.append(news_index[click[i]])
        click = tclick

        if len(click) > MAX_ALL:
            click = click[-MAX_ALL:]
        else:
            click = [0] * (MAX_ALL - len(click)) + click

        user['click'][user_id] = np.array(click)
    return user


def get_train_input(news_index, session):
    sess_pos = []
    sess_neg = []
    user_id = []
    for sess_id in range(len(session)):
        sess = session[sess_id]
        _, poss, negs = sess
        for i in range(len(poss)):
            pos = poss[i]
            neg = newsample(negs, npratio)
            sess_pos.append(pos)
            sess_neg.append(neg)
            user_id.append(sess_id)

    sess_all = np.zeros((len(sess_pos), 1 + npratio), dtype='int32')
    label = np.zeros((len(sess_pos), 1 + npratio))
    for sess_id in range(sess_all.shape[0]):
        pos = sess_pos[sess_id]
        negs = sess_neg[sess_id]
        sess_all[sess_id, 0] = news_index[pos]
        index = 1
        for neg in negs:
            sess_all[sess_id, index] = news_index[neg]
            index += 1
        label[sess_id, 0] = 1
    user_id = np.array(user_id, dtype='int32')

    return sess_all, user_id, label


def get_test_input(news_index, session):
    DocIds = []
    UserIds = []
    Labels = []
    Bound = []
    count = 0
    for sess_id in range(len(session)):
        _, poss, negs = session[sess_id]
        imp = {'labels': [],
               'docs': []}
        start = count
        for i in range(len(poss)):
            docid = news_index[poss[i]]
            DocIds.append(docid)
            Labels.append(1)
            UserIds.append(sess_id)
            count += 1
        for i in range(len(negs)):
            docid = news_index[negs[i]]
            DocIds.append(docid)
            Labels.append(0)
            UserIds.append(sess_id)
            count += 1
        Bound.append([start, count])

    DocIds = np.array(DocIds, dtype='int32')
    UserIds = np.array(UserIds, dtype='int32')
    Labels = np.array(Labels, dtype='float32')

    return DocIds, UserIds, Labels, Bound
