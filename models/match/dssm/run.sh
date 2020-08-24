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
#!/bin/bash
echo "................run................."
python -m paddlerec.run -m ./config.yaml >result1.txt
grep -i "query_doc_sim" ./result1.txt >./result2.txt
sed '$d' result2.txt >result.txt
rm -f result1.txt
rm -f result2.txt
python transform.py
sort -t $'\t' -k1,1 -k 2nr,2 pair.txt >result.txt
rm -f pair.txt
python ../../../tools/cal_pos_neg.py result.txt
