#!/bin/bash

###################################################
# Usage: submit.sh
# Description: run mpi submit clinet implement
###################################################

echo "heheda"

python -m paddlerec.run -m paddlerec.models.rank.dnn -e cluster -r worker
