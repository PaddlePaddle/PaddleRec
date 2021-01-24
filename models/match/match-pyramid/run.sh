#!/bin/bash
echo "................run................."
python -u ../../../tools/trainer.py -m config_bigdata.yaml
python -u ../../../tools/infer.py -m config_bigdata.yaml &>result.txt
python eval.py
