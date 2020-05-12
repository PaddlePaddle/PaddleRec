#!/bin/bash

###################################################
# Usage: submit.sh
# Description: run mpi submit clinet implement
###################################################

#-----------------------------------------------------------------------------------------------------------------
#fun : get argument from env, set it into variables
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function vars_get_from_env() {
  echo "xx"
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
  echo "before_submit"
}

#-----------------------------------------------------------------------------------------------------------------
#fun : after hook submit to cluster
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function after_submit() {
  echo "after_submit"
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
    --hdfs ${g_hdfs_path} \
    --ugi ${g_hdfs_ugi} \
    --hout ${g_hdfs_output} \
    --files ${g_submit_package} \
    -l nodes=${g_job_nodes},walltime=1000:00:00,resource=full ${g_job_entry}

  after_submit
}

function main() {
  echo "run submit done"
}
