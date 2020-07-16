#!/bin/bash
echo "................run................."
python -m paddlerec.run -m ./config.yaml >result1.txt
grep -A1 "prediction" ./result1.txt >./result.txt
rm -f result1.txt
python eval.py
