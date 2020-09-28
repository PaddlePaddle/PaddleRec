# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:20:12 2015

@author: Balázs Hidasi
"""

import numpy as np
import pandas as pd
import datetime as dt
import time

PATH_TO_ORIGINAL_DATA = './'
PATH_TO_PROCESSED_DATA = './'

data = pd.read_csv(
    PATH_TO_ORIGINAL_DATA + 'yoochoose-clicks.dat',
    sep=',',
    header=0,
    usecols=[0, 1, 2],
    dtype={0: np.int32,
           1: str,
           2: np.int64})
data.columns = ['session_id', 'timestamp', 'item_id']
data['Time'] = data.timestamp.apply(lambda x: time.mktime(dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timetuple())) #This is not UTC. It does not really matter.
del (data['timestamp'])

session_lengths = data.groupby('session_id').size()
data = data[np.in1d(data.session_id, session_lengths[session_lengths > 1]
                    .index)]
item_supports = data.groupby('item_id').size()
data = data[np.in1d(data.item_id, item_supports[item_supports >= 5].index)]
session_lengths = data.groupby('session_id').size()
data = data[np.in1d(data.session_id, session_lengths[session_lengths >= 2]
                    .index)]

tmax = data.Time.max()
session_max_times = data.groupby('session_id').Time.max()
session_train = session_max_times[session_max_times < tmax - 86400].index
session_test = session_max_times[session_max_times >= tmax - 86400].index
train = data[np.in1d(data.session_id, session_train)]
test = data[np.in1d(data.session_id, session_test)]
test = test[np.in1d(test.item_id, train.item_id)]
tslength = test.groupby('session_id').size()
test = test[np.in1d(test.session_id, tslength[tslength >= 2].index)]
print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(
    len(train), train.session_id.nunique(), train.item_id.nunique()))
train.to_csv(
    PATH_TO_PROCESSED_DATA + 'rsc15_train_full.txt', sep='\t', index=False)
print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(
    len(test), test.session_id.nunique(), test.item_id.nunique()))
test.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_test.txt', sep='\t', index=False)

tmax = train.Time.max()
session_max_times = train.groupby('session_id').Time.max()
session_train = session_max_times[session_max_times < tmax - 86400].index
session_valid = session_max_times[session_max_times >= tmax - 86400].index
train_tr = train[np.in1d(train.session_id, session_train)]
valid = train[np.in1d(train.session_id, session_valid)]
valid = valid[np.in1d(valid.item_id, train_tr.item_id)]
tslength = valid.groupby('session_id').size()
valid = valid[np.in1d(valid.session_id, tslength[tslength >= 2].index)]
print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(
    len(train_tr), train_tr.session_id.nunique(), train_tr.item_id.nunique()))
train_tr.to_csv(
    PATH_TO_PROCESSED_DATA + 'rsc15_train_tr.txt', sep='\t', index=False)
print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(
    len(valid), valid.session_id.nunique(), valid.item_id.nunique()))
valid.to_csv(
    PATH_TO_PROCESSED_DATA + 'rsc15_train_valid.txt', sep='\t', index=False)
