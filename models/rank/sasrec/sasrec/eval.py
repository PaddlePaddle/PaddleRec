import random
import copy
import os
from tqdm import tqdm

import paddle
import numpy as np


def evaluate(dataset, model, epoch_train, batch_train, args, is_val=True):
    model.eval()
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    before = train
    now = valid if is_val else test

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in tqdm(users):
        if len(before[u]) < 1 or len(now[u]) < 1:
            continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if not is_val:
            seq[idx] = valid[u][0]
            idx -= 1
        for i in reversed(before[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(before[u])
        rated.add(0)
        item_idx = [now[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        predictions = -model.predict(*[paddle.to_tensor(l) for l in [[seq], item_idx]])
        predictions = predictions[0]  # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    NDCG /= valid_user
    HT /= valid_user

    model.train()
    print('\nEpoch {} Evaluation - NDCG: {:.4f}  HIT@10: {:.4f}'.format(epoch_train,  NDCG, HT))
    if args.log_result and is_val:
        with open(os.path.join(args.save_folder, 'result.csv'), 'a') as r:
            r.write('\n{:d},{:d},{:.4f},{:.4f}'.format(epoch_train, batch_train, NDCG, HT))
    return (HT, NDCG)
