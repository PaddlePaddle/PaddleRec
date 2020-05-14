#!/bin/bash

###################################################
# Usage: job.sh
# Description: run mpi job clinet implement
###################################################


# ---------------------------------------------------------------------------- #
#                            variable define                                   #
# ---------------------------------------------------------------------------- #
export CPU_NUM=16
export GLOG_v=0
export FLAGS_rpc_deadline=300000
# ---------------------------------------------------------------------------- #

python -m paddlerec.run -m paddle_rec_config.yaml -e cluster -r worker
