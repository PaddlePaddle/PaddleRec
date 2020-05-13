#! /bin/bash

set -e
echo "begin to prepare data"

mkdir -p data/train
mkdir -p data/test

python generate_synthetic_data.py 

