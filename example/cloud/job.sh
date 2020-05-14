#!/bin/bash

###################################################
# Usage: submit.sh
# Description: run mpi submit clinet implement
###################################################

echo "heheda"

python -m paddlerec.run -m paddle_rec_config.yaml -e cluster -r worker
