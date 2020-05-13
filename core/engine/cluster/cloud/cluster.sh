#!/bin/bash

###################################################
# Usage: submit.sh
# Description: run mpi submit clinet implement
###################################################

# ---------------------------------------------------------------------------- #
#                            variable define                                   #
# ---------------------------------------------------------------------------- #
declare g_jobname=""
declare g_version=""
declare g_priority=""
declare g_nodes=""
declare g_run_cmd=""
declare g_groupname=""
declare g_config=""
declare g_submitfiles=""
declare g_ak=""
declare g_sk=""
declare g_user_define_script=""
# ---------------------------------------------------------------------------- #

#-----------------------------------------------------------------------------------------------------------------
#fun : package
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function package_hook() {
  g_run_stage="package"
  package
}

#-----------------------------------------------------------------------------------------------------------------
#fun : before hook submit to cluster
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function _before_submit() {
  echo "before_submit"
  before_submit_hook
}

#-----------------------------------------------------------------------------------------------------------------
#fun : after hook submit to cluster
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function _after_submit() {
  echo "after_submit"
  after_submit_hook
}

#-----------------------------------------------------------------------------------------------------------------
#fun : submit to cluster
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function _submit() {
  g_run_stage="submit"

  paddlecloud job --ak ${g_ak} --sk ${g_sk} train --cluster-name ${g_jobname} \
    --job-version ${g_version} \
    --mpi-priority ${g_priority} \
    --mpi-wall-time 300:59:00 \
    --mpi-nodes ${g_nodes} --is-standalone 0 \
    --mpi-memory 110Gi \
    --job-name ${g_jobname} \
    --start-cmd ${g_run_cmd} \
    --group-name ${g_groupname} \
    --job-conf ${g_config} \
    --files ${g_submitfiles} \
    --json
}

function submit_hook() {
  _before_submit

  _submit

  _after_submit
}

function main() {
  source ${g_user_define_script}
  package_hook
  submit_hook
}

main