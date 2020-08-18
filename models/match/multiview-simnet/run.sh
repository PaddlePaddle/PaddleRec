#!/bin/bash
echo "................run................."
python -m paddlerec.run -m ./config.yaml >result1.txt
grep -i "query_pt_sim" ./result1.txt >./result2.txt
sed '$d' result2.txt >result.txt
rm -f result1.txt
rm -f result2.txt
python eval.py
