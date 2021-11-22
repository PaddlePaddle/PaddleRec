import sys
import copy
import random
import numpy as np
from multiprocessing import Process, Queue


# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue=None, SEED=42):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)  # TODO

    if result_queue is None:
        np.random.seed(SEED)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())
        return zip(*one_batch)
    else:
        np.random.seed(SEED)
        while True:
            one_batch = []
            for i in range(batch_size):
                one_batch.append(sample())

            result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.n_workers = n_workers
        if self.n_workers != 0:
            self.result_queue = Queue(maxsize=n_workers * 10)
            self.processors = []
            for i in range(n_workers):
                self.processors.append(
                    Process(target=sample_function, args=(User,
                                                          usernum,
                                                          itemnum,
                                                          batch_size,
                                                          maxlen,
                                                          self.result_queue,
                                                          np.random.randint(2e9)
                                                          )))
                self.processors[-1].daemon = True
                self.processors[-1].start()
        else:
            self.User = User
            self.usernum = usernum
            self.itemnum = itemnum
            self.batch_size = batch_size
            self.maxlen = maxlen

    def next_batch(self):
        if self.n_workers != 0:
            return self.result_queue.get()
        return sample_function(self.User,
                               self.usernum,
                               self.itemnum,
                               self.batch_size,
                               self.maxlen,
                               None,
                               np.random.randint(2e9))

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
