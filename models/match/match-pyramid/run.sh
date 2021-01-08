#!/bin/bash
echo "................run................."
python -m paddlerec.run -m ./config_bigdata.yaml &>result.txt
python eval.py
