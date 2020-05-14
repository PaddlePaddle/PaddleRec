#!/bin/bash

###################################################
# Usage: submit.sh
# Description: run mpi submit clinet implement
###################################################

g_package_files=""

#-----------------------------------------------------------------------------------------------------------------
#fun : before hook submit to cluster
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function before_submit_hook() {
  echo "before_submit"
}

#-----------------------------------------------------------------------------------------------------------------
#fun : after hook submit to cluster
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function after_submit_hook() {
  echo "after_submit"
}

#-----------------------------------------------------------------------------------------------------------------
#fun : package to cluster
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function package() {
  echo "package"
  temp=${engine_temp_path}

  cp ${engine_workspace}/job.sh ${temp}
  cp ${engine_workspace}/before_hook.sh ${temp}
  cp ${engine_run_config} ${temp}/paddle_rec_config.yaml

  g_submitfiles="job.sh before_hook.sh paddle_rec_config.yaml"
  g_run_cmd="sh job.sh"
}
