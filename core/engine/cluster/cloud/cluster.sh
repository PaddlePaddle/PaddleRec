#!/bin/bash
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


###################################################
# Usage: submit.sh
# Description: run mpi submit client implement
###################################################

# ---------------------------------------------------------------------------- #
#                            variable define                                   #
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

  cd ${engine_temp_path}

  paddlecloud job --ak ${engine_submit_ak} --sk ${engine_submit_sk} train --cluster-name ${engine_submit_cluster} \
    --job-version ${engine_submit_version} \
    --mpi-priority ${engine_submit_priority} \
    --mpi-wall-time 300:59:00 \
    --mpi-nodes ${engine_submit_nodes} --is-standalone 0 \
    --mpi-memory 110Gi \
    --job-name ${engine_submit_jobname} \
    --start-cmd "${g_run_cmd}" \
    --group-name ${engine_submit_group} \
    --job-conf ${engine_submit_config} \
    --files ${g_submitfiles} \
    --json

  cd -
}

function submit_hook() {
  _before_submit
  _submit
  _after_submit
}

function main() {
  source ${engine_submit_scrpit}

  package_hook
  submit_hook
}

main
