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
}

function user_define_variables() {
  echo "user_define_variables"
  g_run_stage="user_define_variables"
}

function job() {
  mpirun -npernode 2 -timestamp-output -tag-output -machinefile ${PBS_NODEFILE} python -u ${g_job_entry}
}

function main() {
  user_define_variables
  env_prepare
  job
}

main
