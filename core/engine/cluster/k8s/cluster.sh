#!/bin/bash

###################################################
# Usage: submit.sh
# Description: run k8s submit client implement
###################################################

# ---------------------------------------------------------------------------- #
#                            variable define                                   #
# ---------------------------------------------------------------------------- #

function create_config_map() {
  echo "Create configmap"
  kubectl delete configmap modelconfig
  kubectl create configmap modelconfig --from-file=${abs_dir}/k8s/set_k8s_env.sh,${paddlerec_model_config}
}

function create_k8s_yaml() {
  echo "Create k8s.yaml"
  rm ${PWD}/k8s.yaml

  let total_pod_num=${engine_submit_trainer_num}+${engine_submit_server_num}
  echo "--K8S ENV-- Job name: ${engine_job_name}"
  echo "--K8S ENV-- Total pod nums: ${total_pod_num}"
  echo "--K8S ENV-- Trainer nums: ${engine_submit_trainer_num}"
  echo "--K8S ENV-- Pserver nums: ${engine_submit_server_num}"
  echo "--K8S ENV-- Docker image: ${engine_submit_docker_image}"
  echo "--K8S ENV-- Threads(cpu_num) ${CPU_NUM}"
  echo "--K8S ENV-- Memory ${engine_submit_memory}"
  echo "--K8S ENV-- Storage ${engine_submit_storage}"
  echo "--K8S ENV-- Log level ${engine_submit_log_level}"
  

  sed -e "s#<$ JOB_NAME $>#$engine_job_name#g" \
      -e "s#<$ TOTAL_POD_NUM $>#$total_pod_num#g" \
      -e "s#<$ TRAINER_NUM $>#$engine_submit_trainer_num#g" \
      -e "s#<$ PSERVER_NUM $>#$engine_submit_server_num#g" \
      -e "s#<$ IMAGE $>#$engine_submit_docker_image#g" \
      -e "s#<$ CPU_NUM $>#$CPU_NUM#g" \
      -e "s#<$ MEMORY $>#$engine_submit_memory#g" \
      -e "s#<$ STORAGE $>#$engine_submit_storage#g" \
      -e "s#<$ GLOG_V $>#$engine_submit_log_level#g" \
      ${abs_dir}/k8s/k8s.yaml.template > ${PWD}/k8s.yaml
}



#-----------------------------------------------------------------------------------------------------------------
#fun : submit to cluster
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function submit() {
  echo "Submit"
  kubectl delete jobs.batch.volcano.sh $engine_job_name
  kubectl apply -f ${PWD}/k8s.yaml
}


function main() {
  create_config_map
  create_k8s_yaml
  submit
}

main
