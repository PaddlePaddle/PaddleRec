#!/bin/bash

###################################################
# Usage: submit.sh
# Description: run k8s submit client implement
###################################################

# ---------------------------------------------------------------------------- #
#                            variable define                                   #
# ---------------------------------------------------------------------------- #

function create_config_map() {
  echo "create_config_map"
  kubectl delete configmap modelconfig
  kubectl create configmap modelconfig --from-file=./set_k8s_env.sh,${paddlerec_model_config}
}

function create_k8s_yaml() {
  echo "create_k8s_yaml"
  rm ${train_workspace}/k8s.yaml

  total_pod_num=${engine_submit_trainer_num}+${engine_submit_server_num}
  echo "Total pod num: ${total_pod_num}"
  
  sed -e "s#<$ JOB_NAME $>#$engine_job_name#g" \
      -e "s#<$ TOTAL_POD_NUM $>#$total_pod_num#g" \
      -e "s#<$ TRAINER_NUM $>#$engine_submit_trainer_num#g" \
      -e "s#<$ SERVER_NUM $>#$engine_submit_server_num#g" \
      -e "s#<$ IMAGE $>#$engine_submit_docker_image#g" \
      ./k8s.yaml.template > ${train_workspace}/k8s.yaml
}



#-----------------------------------------------------------------------------------------------------------------
#fun : submit to cluster
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function submit() {
  echo "submit"
  kubectl apply -f ${train_workspace}/k8s.yaml
}


function main() {
  create_config_map
  create_k8s_yaml
  submit
}

main
