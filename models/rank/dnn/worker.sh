#!/bin/bash

###################################################
# Usage: job.sh
# Description: run job on mpi per node
###################################################

# ---------------------------------------------------------------------------- #
#                            variable define                                   #
# ---------------------------------------------------------------------------- #
declare g_curPath=""
declare g_scriptName=""
declare g_workPath=""
declare g_run_stage=""

# ---------------------------------------------------------------------------- #
#                             const define                                     #
# ---------------------------------------------------------------------------- #
declare -r FLAGS_communicator_thread_pool_size=5
declare -r FLAGS_communicator_send_queue_size=18
declare -r FLAGS_communicator_thread_pool_size=20
declare -r FLAGS_communicator_max_merge_var_num=18
################################################################################

#-----------------------------------------------------------------------------------------------------------------
#fun : check function return code
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function check_error() {
  if [ ${?} -ne 0 ]; then
    echo "execute " + $g_run_stage + " raise exception! please check ..."
    exit 1
  fi
}

#-----------------------------------------------------------------------------------------------------------------
#fun : check function return code
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function env_prepare() {
  g_run_stage="env_prepare"
  WORKDIR=$(pwd)
  mpirun -npernode 1 mv package/* ./
  echo "current:"$WORKDIR
  export LIBRARY_PATH=$WORKDIR/python/lib:$LIBRARY_PATH

  mpirun -npernode 1 python/bin/python -m pip install whl/fleet_rec-0.0.2-py2-none-any.whl --index-url=http://pip.baidu.com/pypi/simple --trusted-host pip.baidu.com >/dev/null
  check_error
}

function run() {
  echo "run"
  g_run_stage="run"
  mpirun -npernode 2 -timestamp-output -tag-output -machinefile ${PBS_NODEFILE} python/bin/python -u -m fleetrec.run -m fleetrec.models.rank.dnn --engine cluster --role worker
}

function main() {
  env_prepare
  run
}

main
