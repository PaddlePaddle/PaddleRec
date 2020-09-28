#!/bin/bash
echo "................run................."
python -m paddlerec.run -m ./config.yaml &>result.txt
python eval.py
