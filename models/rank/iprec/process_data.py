import pandas as pd
from joblib import Parallel, delayed
import multiprocessing, time
import copy

data = pd.read_csv('3day.csv', sep='\t', keep_default_na=False, header=None, error_bad_lines=False)

f_data = pd.DataFrame()
tra_data = pd.DataFrame()
val_data = pd.DataFrame()
tes_data = pd.DataFrame()
u_list, s_u_list, pos_list, neg_list = [], [], [], []


def fun_avg(user, d):
    neg_ratio = 5
    u_list.append(user)
    pos = d[d[3] == 1]
    neg = d[d[3] == 0]
    l1 = pos.shape[0]
    l2 = neg.shape[0]
    assert l1 + l2 == d.shape[0]
    pos = pos.reset_index(drop=True)
    neg = neg.reset_index(drop=True)
    if l1 >= 3 and not neg.empty:
        s_u_list.append(user)
        pos_list.append(l1)
        neg_list.append(l2)
        #    sample train.validation.test
        if l2 >= neg_ratio * l1:
            s_neg = neg[:neg_ratio * l1]
        else:
            s_neg = neg
        s_data = pd.concat([pos, s_neg], ignore_index=True).sample(frac=1).reset_index(drop=True)
        tra_data = s_data[:int(s_data.shape[0] * 0.7)]
        val_data = s_data[int(s_data.shape[0] * 0.7):int(s_data.shape[0] * 0.8)]
        tes_data = s_data[int(s_data.shape[0] * 0.8):]
        return s_data, tra_data, val_data, tes_data


def applyParallel(dfGrouped, func):
    res = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(name, group) for name, group in dfGrouped)
    res = list(filter(None, res))
    #     print(res)
    d = [item[0] for item in res]
    t = [item[1] for item in res]
    v = [item[2] for item in res]
    s = [item[3] for item in res]
    return pd.concat(d, ignore_index=True), pd.concat(t, ignore_index=True), pd.concat(v, ignore_index=True), pd.concat(
        s, ignore_index=True)


t1 = time.time()
f_data, tra_data, val_data, tes_data = applyParallel(data.groupby(0), fun_avg)
print(time.time() - t1)
f_max_len = 20
user_set = set(tra_data[0])
item_set = set(tra_data[2])
biz_set = set(tra_data[4])
user_set.update(
    [item for sublist in tra_data[6] for item in list(map(lambda x: int(x), sublist.split(',')[:f_max_len]))])
user_map = dict(zip(user_set, range(1, len(user_set) + 1)))
item_map = dict(zip(item_set, range(1, len(item_set) + 1)))
biz_map = dict(zip(biz_set, range(1, len(biz_set) + 1)))
tes_data = tes_data[(tes_data[2].isin(item_set)) & (tes_data[4].isin(biz_set))]
tes_data[6] = tes_data[6].map(lambda x: ','.join([y for y in x.split(',')[:f_max_len] if int(y) in user_set]))
tes_data = tes_data[tes_data[6] != '']
val_data = val_data[(val_data[2].isin(item_set)) & (val_data[4].isin(biz_set))]
val_data[6] = val_data[6].map(lambda x: ','.join([y for y in x.split(',')[:f_max_len] if int(y) in user_set]))
val_data = val_data[val_data[6] != '']
tra_data[0] = tra_data[0].map(lambda x: user_map[x])
tra_data[2] = tra_data[2].map(lambda x: item_map[x])
tra_data[4] = tra_data[4].map(lambda x: biz_map[x])
tra_data[6] = tra_data[6].map(lambda x: [user_map[int(y)] for y in x.split(',')[:f_max_len]])
tra_data = tra_data.drop([1], axis=1)
val_data[0] = val_data[0].map(lambda x: user_map[x])
val_data[2] = val_data[2].map(lambda x: item_map[x])
val_data[4] = val_data[4].map(lambda x: biz_map[x])
val_data[6] = val_data[6].map(lambda x: [user_map[int(y)] for y in x.split(',')[:f_max_len]])
val_data = val_data.drop([1], axis=1)
tes_data[0] = tes_data[0].map(lambda x: user_map[x])
tes_data[2] = tes_data[2].map(lambda x: item_map[x])
tes_data[4] = tes_data[4].map(lambda x: biz_map[x])
tes_data[6] = tes_data[6].map(lambda x: [user_map[int(y)] for y in x.split(',')[:f_max_len]])
tes_data = tes_data.drop([1], axis=1)
f_data = pd.concat([tra_data, val_data, tes_data], ignore_index=True)
from collections import defaultdict

user_packages = dict()
user_items = dict()
user_bizs = dict()
user_friends = dict()
pack_neighbors_b = defaultdict(list)
pack_neighbors_f = defaultdict(list)

t1 = time.time()
for u, d in f_data.groupby([0]):
    # def fun_avg1(u,d):
    #     global pack_neighbors_f
    i = list(d[2])
    b = list(d[4])
    f = list(map(lambda x: x + [0] * (f_max_len - len(x)), d[6]))
    packages = list(zip(i, b, f))
    for index in range(len(packages)):
        for j in range(index + 1, len(packages)):
            a = set(packages[index][2]) & set(packages[j][2]) - {0}
            if a and packages[index][0] != packages[j][0]:
                pack_neighbors_f[str(u) + '_' + str(packages[index][0])].append((len(a), packages[j]))
                pack_neighbors_f[str(u) + '_' + str(packages[j][0])].append((len(a), packages[index]))
    for item in set(i):
        try:
            pack_neighbors_f[str(u) + '_' + str(item)].sort(key=lambda x: x[0], reverse=True)
            pack_neighbors_f[str(u) + '_' + str(item)] = list(
                map(lambda x: x[1], pack_neighbors_f[str(u) + '_' + str(item)]))
        except:
            print(pack_neighbors_f[str(u) + '_' + str(packages[index][0])])
print(time.time() - t1)

t1 = time.time()
for name, d in f_data.groupby([0, 4]):
    # def fun_avg2(name,d):
    #     global pack_neighbors_b
    if d.shape[0] > 1:
        i = list(d[2])
        b = list(d[4])
        f = list(map(lambda x: x + [0] * (f_max_len - len(x)), d[6]))
        packages = list(zip(i, b, f))
        for index in range(len(packages)):
            tmp = packages.copy()
            del tmp[index]
            #             print(name[0],packages[index][0])
            pack_neighbors_b[str(name[0]) + '_' + str(packages[index][0])] = tmp
print(time.time() - t1)

t1 = time.time()
for user, d in tra_data.groupby(0):
    # def fun_avg3(user,d):
    #     global user_packages
    pos = d[(d[3] == 1)]
    b_pos = d[(d[5] == 1)]
    i = list(pos[2])
    b = list(pos[4])
    f = list(map(lambda x: x + [0] * (f_max_len - len(x)), pos[6]))
    f_ = list(pos[6])
    b_ = list(b_pos[4])
    user_packages[user] = list(zip(i, b, f))
    user_items[user] = i
    user_bizs[user] = b_
    user_friends[user] = [item for sublist in f_ for item in sublist]
print(time.time() - t1)
tra_data = tra_data.sample(frac=1).reset_index(drop=True)
from tqdm import tqdm
import json

int_list = lambda x: [int(y) for y in x]


def to_jsonl1(features, file):
    writer = open(file, 'w+')
    size = features.shape[0]
    for i in tqdm(range(size)):
        pack = str(features.iloc[i][0]) + '_' + str(features.iloc[i][2])
        u_i = copy.deepcopy(user_items[features.iloc[i][0]])
        u_b = copy.deepcopy(user_bizs[features.iloc[i][0]])
        u_f = copy.deepcopy(user_friends[features.iloc[i][0]])
        #         print(features.iloc[i][2],features.iloc[i][4],features.iloc[i][6])
        #         print(u_i,u_b,u_f)
        u_p = copy.deepcopy(user_packages[features.iloc[i][0]])
        #         print(u_p)
        if features.iloc[i][3] == 1:
            u_i.remove(features.iloc[i][2])
            #             u_b.remove(features.iloc[i][4])
            for x in features.iloc[i][6]:
                u_f.remove(x)
            #             print('-------',u_i,u_b,u_f)
            for j, x in enumerate(u_p):
                if x[0] == features.iloc[i][2]:
                    del u_p[j]
                    break
        #             print('-------------------')
        #             print(features.iloc[i])
        #             print(u_p)
        example = {
            "user": int(features.iloc[i][0]),
            "item": int(features.iloc[i][2]),
            "biz": int(features.iloc[i][4]),
            "friends": features.iloc[i][6],
            "user_items": u_i,
            "user_bizs": u_b,
            "user_friends": u_f,
            "user_packages": to_list(u_p[:50]),
            "pack_neighbors_b": to_list(pack_neighbors_b[pack][:20]),
            "pack_neighbors_f": to_list(pack_neighbors_f[pack][:20]),
            "label": int(features.iloc[i][3]),
            "label2": int(features.iloc[i][5])
        }

        writer.write(json.dumps(example, ensure_ascii=False))
        writer.write('\n')

    writer.close()


def to_list(l):
    res = []
    for i in l:
        res.append(i[0])
        res.append(i[1])
        res.extend(i[2])
    return res


def to_jsonl(features, file):
    writer = open(file, 'w+')
    size = features.shape[0]
    for i in tqdm(range(size)):
        pack = str(features.iloc[i][0]) + '_' + str(features.iloc[i][2])
        example = {
            "user": int(features.iloc[i][0]),
            "item": int(features.iloc[i][2]),
            "biz": int(features.iloc[i][4]),
            "friends": int_list(features.iloc[i][6]),
            "user_items": int_list(user_items[features.iloc[i][0]][:50]),
            "user_bizs": int_list(user_bizs[features.iloc[i][0]][:50]),
            "user_friends": int_list(user_friends[features.iloc[i][0]][:50 * f_max_len]),
            "user_packages": int_list(to_list(user_packages[features.iloc[i][0]][:50])),
            "pack_neighbors_b": int_list(to_list(pack_neighbors_b[pack][:20])),
            "pack_neighbors_f": int_list(to_list(pack_neighbors_f[pack][:20])),
            "label": int(features.iloc[i][3]),
            "label2": int(features.iloc[i][5])
        }
        writer.write(json.dumps(example, ensure_ascii=False))
        writer.write('\n')
    writer.close()


to_jsonl(tes_data, 'test.jsonl')
to_jsonl1(tra_data, 'train.jsonl')
to_jsonl(val_data, 'val.jsonl')
