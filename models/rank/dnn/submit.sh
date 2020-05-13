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

  temp=${engine_temp_path}
  echo "package temp dir: " ${temp}

  cp ${engine_job_scrpit} ${temp}
  cp ${engine_submit_qconf} ${temp}
  echo "copy job.sh from " ${engine_worker} " to " ${temp}

  mkdir -p ${temp}/package
  cp -r ${engine_package_python} ${temp}/package/
  echo "copy python from " ${engine_package_python} " to " ${temp}

  mkdir ${temp}/package/whl
  cp ${engine_package_paddlerec} ${temp}/package/whl/
  echo "copy " ${engine_package_paddlerec} " to " ${temp}"/whl/"
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
  g_job_name="paddle_rec_mpi"
  g_hdfs_path=$g_hdfs_path
  g_job_entry="worker.sh"

  engine_hdfs_output=${engine_hdfs_output}/$(date +%Y%m%d%H%M%S)

  cd ${engine_temp_path}

  ${engine_submit_hpc}/bin/qsub_f \
    -N ${g_job_name} \
    --conf ${engine_submit_qconf} \
    --hdfs ${engine_hdfs_name} \
    --ugi ${engine_hdfs_ugi} \
    --hout ${engine_hdfs_output} \
    --files ./package \
    -l nodes=${engine_submit_nodes},walltime=1000:00:00,resource=full ${g_job_entry}
}

function main() {
  package

  before_submit
  submit
  after_submit
}
