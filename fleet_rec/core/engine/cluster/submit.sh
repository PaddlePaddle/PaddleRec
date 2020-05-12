#!/bin/bash

###################################################
# Usage: submit.sh
# Description: run mpi submit clinet
###################################################

# ---------------------------------------------------------------------------- #
#                            variable define                                   #
# ---------------------------------------------------------------------------- #
declare g_curPath=""
declare g_scriptName=""
declare g_workPath=""
declare g_run_stage=""

# ----------------------------for hpc submit  -------------------------------- #
declare g_hpc_path=""
declare g_job_name=""
declare g_qsub_conf=""
declare g_hdfs_path=""
declare g_hdfs_ugi=""
declare g_hdfs_output=""
declare g_submit_package=""
declare g_job_nodes=""
declare g_job_entry=""

# ---------------------------------------------------------------------------- #
#                             const define                                     #
# ---------------------------------------------------------------------------- #
declare -r CALL="x"
################################################################################


#-----------------------------------------------------------------------------------------------------------------
# Function: get_cur_path
# Description: get churrent path
# Parameter:
#   input:
#   N/A
#   output:
#   N/A
# Return: 0 -- success; not 0 -- failure
# Others: N/A
#-----------------------------------------------------------------------------------------------------------------
get_cur_path()
{
  g_run_stage="get_cur_path"
    cd "$(dirname "${BASH_SOURCE-$0}")"
    g_curPath="${PWD}"
    g_scriptName="$(basename "${BASH_SOURCE-$0}")"
    cd - >/dev/null
}


#-----------------------------------------------------------------------------------------------------------------
#fun : get argument from env, set it into variables
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function vars_get_from_env() {
    g_run_stage="vars_get_from_env"
    g_hpc_path=${engine.}
    g_crontabDate=$2
}


#-----------------------------------------------------------------------------------------------------------------
#fun : check function return code
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function check_error()
{
    if [ ${?} -ne 0 ]
    then
        echo "execute " + $g_run_stage +  " raise exception! please check ..."
        exit 1
    fi
}


#-----------------------------------------------------------------------------------------------------------------
#fun : package
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function package() {
  g_run_stage="package"

}


#-----------------------------------------------------------------------------------------------------------------
#fun : before hook submit to cluster
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function before_submit() {

}


#-----------------------------------------------------------------------------------------------------------------
#fun : after hook submit to cluster
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function after_submit() {

}


#-----------------------------------------------------------------------------------------------------------------
#fun : submit to cluster
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function submit() {
  g_run_stage="submit"

  before_submit

  ${g_hpc_path}/bin/qsub_f \
      -N ${g_job_name} \
      --conf ${g_qsub_conf} \
      --hdfs ${g_hdfs_path}  \
      --ugi ${g_hdfs_ugi} \
      --hout ${g_hdfs_output} \
      --files ${g_submit_package} \
      -l nodes=${g_job_nodes},walltime=1000:00:00,resource=full ${g_job_entry}

  after_submit
}

function main() {
  get_cur_path
  check_error

  vars_get_from_env
  check_error

  package
  check_error

  submit
  check_error
}

main
