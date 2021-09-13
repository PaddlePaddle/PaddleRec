# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from sklearn.metrics import f1_score
import numpy as np


def topk_f1_score(labels,
                  probs,
                  topk_list=None,
                  average="macro",
                  threshold=None):
    assert topk_list is not None or threshold is not None, "one of topklist and threshold should not be None"
    if threshold is not None:
        preds = probs > threshold
    else:
        preds = np.zeros_like(labels, dtype=np.int64)
        for idx, (prob, topk) in enumerate(zip(np.argsort(probs), topk_list)):
            preds[idx][prob[-int(topk):]] = 1
    return f1_score(labels, preds, average=average)


fetch_batch_var = np.load(
    "output_model_multi_class/result.npy", allow_pickle=True)
labels = fetch_batch_var[1]
probs = fetch_batch_var[2]
topk = fetch_batch_var[3]
test_macro_f1 = topk_f1_score(labels, probs, topk, "macro", 0.3)
print("Macro", test_macro_f1)
