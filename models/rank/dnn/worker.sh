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
export FLAGS_communicator_thread_pool_size=5
export FLAGS_communicator_send_queue_size=18
export FLAGS_communicator_thread_pool_size=20
export FLAGS_communicator_max_merge_var_num=18
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

  mpirun -npernode 1 tar -zxvf python.tar.gz > /dev/null

  export PYTHONPATH=$WORKDIR/python/
  export PYTHONROOT=$WORKDIR/python/
  export LIBRARY_PATH=$PYTHONPATH/lib:$LIBRARY_PATH
  export LD_LIBRARY_PATH=$PYTHONPATH/lib:$LD_LIBRARY_PATH
  export PATH=$PYTHONPATH/bin:$PATH
  export LIBRARY_PATH=$PYTHONROOT/lib:$LIBRARY_PATH

  python -c "print('heheda')"

  mpirun -npernode 1 python/bin/python -m pip uninstall -y paddle-rec
  mpirun -npernode 1 python/bin/python -m pip install whl/fleet_rec-0.0.2-py2-none-any.whl --index-url=http://pip.baidu.com/pypi/simple --trusted-host pip.baidu.com
  check_error
}

function run() {
  echo "run"
  g_run_stage="run"
  mpirun -npernode 2 -timestamp-output -tag-output -machinefile ${PBS_NODEFILE} python/bin/python -u -m paddlerec.run -m paddlerec.models.rank.dnn --engine cluster --role worker
}

function main() {
  env_prepare
  run
}

main
