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
# Description: run paddlecloud submit client implement
###################################################

# ---------------------------------------------------------------------------- #
#                            variable define                                   #
# ---------------------------------------------------------------------------- #

#-----------------------------------------------------------------------------------------------------------------
#fun : before hook submit to cluster
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function _before_submit() {
  echo "before_submit"
  
  if [ ${DISTRIBUTE_MODE} == "PS_CPU_MPI" ]; then
    _gen_cpu_before_hook
    _gen_mpi_config
    _gen_mpi_job
    _gen_end_hook
  elif [ ${DISTRIBUTE_MODE} == "COLLECTIVE_GPU_K8S" ]; then
    _gen_gpu_before_hook
    _gen_k8s_config
    _gen_k8s_gpu_job
    _gen_end_hook
  elif [ ${DISTRIBUTE_MODE} == "PS_CPU_K8S" ]; then
    _gen_cpu_before_hook
    _gen_k8s_config
    _gen_k8s_cpu_job
    _gen_end_hook
  fi
  
}

function _gen_mpi_config() {
  echo "gen mpi_config.ini"
  sed -e "s#<$ FS_NAME $>#$FS_NAME#g" \
      -e "s#<$ FS_UGI $>#$FS_UGI#g" \
      -e "s#<$ TRAIN_DATA_PATH $>#$TRAIN_DATA_PATH#g" \
      -e "s#<$ TEST_DATA_PATH $>#$TEST_DATA_PATH#g" \
      -e "s#<$ OUTPUT_PATH $>#$OUTPUT_PATH#g" \
      -e "s#<$ THIRDPARTY_PATH $>#$THIRDPARTY_PATH#g" \
      -e "s#<$ CPU_NUM $>#$max_thread_num#g" \
      -e "s#<$ FLAGS_communicator_is_sgd_optimizer $>#$FLAGS_communicator_is_sgd_optimizer#g" \
      -e "s#<$ FLAGS_communicator_send_queue_size $>#$FLAGS_communicator_send_queue_size#g" \
      -e "s#<$ FLAGS_communicator_thread_pool_size $>#$FLAGS_communicator_thread_pool_size#g" \
      -e "s#<$ FLAGS_communicator_max_merge_var_num $>#$FLAGS_communicator_max_merge_var_num#g" \
      -e "s#<$ FLAGS_communicator_max_send_grad_num_before_recv $>#$FLAGS_communicator_max_send_grad_num_before_recv#g" \
      -e "s#<$ FLAGS_communicator_fake_rpc $>#$FLAGS_communicator_fake_rpc#g" \
      -e "s#<$ FLAGS_rpc_retry_times $>#$FLAGS_rpc_retry_times#g" \
      ${abs_dir}/cloud/mpi_config.ini.template >${PWD}/config.ini
}

function _gen_k8s_config() {
  echo "gen k8s_config.ini"
  sed -e "s#<$ FS_NAME $>#$FS_NAME#g" \
      -e "s#<$ FS_UGI $>#$FS_UGI#g" \
      -e "s#<$ AFS_REMOTE_MOUNT_POINT $>#$AFS_REMOTE_MOUNT_POINT#g" \
      -e "s#<$ OUTPUT_PATH $>#$OUTPUT_PATH#g" \
      -e "s#<$ CPU_NUM $>#$max_thread_num#g" \
      -e "s#<$ FLAGS_communicator_is_sgd_optimizer $>#$FLAGS_communicator_is_sgd_optimizer#g" \
      -e "s#<$ FLAGS_communicator_send_queue_size $>#$FLAGS_communicator_send_queue_size#g" \
      -e "s#<$ FLAGS_communicator_thread_pool_size $>#$FLAGS_communicator_thread_pool_size#g" \
      -e "s#<$ FLAGS_communicator_max_merge_var_num $>#$FLAGS_communicator_max_merge_var_num#g" \
      -e "s#<$ FLAGS_communicator_max_send_grad_num_before_recv $>#$FLAGS_communicator_max_send_grad_num_before_recv#g" \
      -e "s#<$ FLAGS_communicator_fake_rpc $>#$FLAGS_communicator_fake_rpc#g" \
      -e "s#<$ FLAGS_rpc_retry_times $>#$FLAGS_rpc_retry_times#g" \
      ${abs_dir}/cloud/k8s_config.ini.template >${PWD}/config.ini
}

function _gen_cpu_before_hook() {
  echo "gen cpu before_hook.sh"
  sed -e "s#<$ PADDLEPADDLE_VERSION $>#$PADDLE_VERSION#g" \
    ${abs_dir}/cloud/before_hook_cpu.sh.template >${PWD}/before_hook.sh
}

function _gen_gpu_before_hook() {
  echo "gen gpu before_hook.sh"
  sed -e "s#<$ PADDLEPADDLE_VERSION $>#$PADDLE_VERSION#g" \
    ${abs_dir}/cloud/before_hook_gpu.sh.template >${PWD}/before_hook.sh
}

function _gen_end_hook() {
  echo "gen end_hook.sh"
  cp ${abs_dir}/cloud/end_hook.sh.template ${PWD}/end_hook.sh
}

function _gen_mpi_job() {
  echo "gen mpi_job.sh"
  sed -e "s#<$ GROUP_NAME $>#$GROUP_NAME#g" \
      -e "s#<$ JOB_NAME $>#$OLD_JOB_NAME#g" \
      -e "s#<$ AK $>#$AK#g" \
      -e "s#<$ SK $>#$SK#g" \
      -e "s#<$ MPI_PRIORITY $>#$PRIORITY#g" \
      -e "s#<$ MPI_NODES $>#$MPI_NODES#g" \
      -e "s#<$ START_CMD $>#$START_CMD#g" \
      ${abs_dir}/cloud/mpi_job.sh.template >${PWD}/job.sh
}

function _gen_k8s_gpu_job() {
  echo "gen k8s_job.sh"
  sed -e "s#<$ GROUP_NAME $>#$GROUP_NAME#g" \
      -e "s#<$ JOB_NAME $>#$OLD_JOB_NAME#g" \
      -e "s#<$ AK $>#$AK#g" \
      -e "s#<$ SK $>#$SK#g" \
      -e "s#<$ K8S_PRIORITY $>#$PRIORITY#g" \
      -e "s#<$ K8S_TRAINERS $>#$K8S_TRAINERS#g" \
      -e "s#<$ K8S_CPU_CORES $>#$K8S_CPU_CORES#g" \
      -e "s#<$ K8S_GPU_CARD $>#$K8S_GPU_CARD#g" \
      -e "s#<$ START_CMD $>#$START_CMD#g" \
      ${abs_dir}/cloud/k8s_job.sh.template >${PWD}/job.sh
}

function _gen_k8s_cpu_job() {
  echo "gen k8s_job.sh"
  sed -e "s#<$ GROUP_NAME $>#$GROUP_NAME#g" \
      -e "s#<$ JOB_NAME $>#$OLD_JOB_NAME#g" \
      -e "s#<$ AK $>#$AK#g" \
      -e "s#<$ SK $>#$SK#g" \
      -e "s#<$ K8S_PRIORITY $>#$PRIORITY#g" \
      -e "s#<$ K8S_TRAINERS $>#$K8S_TRAINERS#g" \
      -e "s#<$ K8S_PS_NUM $>#$K8S_PS_NUM#g" \
      -e "s#<$ K8S_PS_CORES $>#$K8S_PS_CORES#g" \
      -e "s#<$ K8S_CPU_CORES $>#$K8S_CPU_CORES#g" \
      -e "s#<$ START_CMD $>#$START_CMD#g" \
      ${abs_dir}/cloud/k8s_cpu_job.sh.template >${PWD}/job.sh
}


#-----------------------------------------------------------------------------------------------------------------
#fun : after hook submit to cluster
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function _after_submit() {
  echo "end submit"
}

#-----------------------------------------------------------------------------------------------------------------
#fun : submit to cluster
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function _submit() {
  g_run_stage="submit"
  sh job.sh
}

function package_hook() {
  cur_time=`date  +"%Y%m%d%H%M"`
  new_job_name="${JOB_NAME}_${cur_time}"
  export OLD_JOB_NAME=${JOB_NAME}
  export JOB_NAME=${new_job_name}
  export job_file_path="${PWD}/${new_job_name}"
  mkdir ${job_file_path}
  cp $FILES ${job_file_path}/
  cd ${job_file_path}
  echo "The task submission folder is generated at ${job_file_path}"
}

function submit_hook() {
  _before_submit
  _submit
  _after_submit
}

function main() {
  package_hook
  submit_hook
}

main
