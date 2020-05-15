import re
import sys
import collections
import os
import six
import time
import numpy as np
import paddle.fluid as fluid
import paddle
import csv
import io

from paddlerec.core.reader import Reader
from paddlerec.core.utils import envs

class TrainReader(Reader):
    def init(self):
        pass

    def _process_line(self, l): 
        l = l.strip().split(" ")
        data = l[0:10]
        seq_len = l[10:11]
        label = l[11:]
        return  data, label, seq_len

    def generate_sample(self, line):
        def data_iter():
            data, label, seq_len = self._process_line(line)
            if data is None:
                yield None
                return
            yield [('data', data), ('label', label), ('seq_len', seq_len)]
        return data_iter
