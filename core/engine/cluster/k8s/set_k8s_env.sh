#!/bin/bash
set -x

check_failed_cnt() {
  max_failed=$1
  failed_count=$(python -m paddlerec.tools.k8s_tools count_pods_by_phase paddle-job=${PADDLE_JOB_NAME} Failed)
  if [ $failed_count -gt $max_failed ]; then
    stdbuf -oL echo "Failed trainer count beyond the threadhold: "$max_failed
    echo "Failed trainer count beyond the threshold: " $max_failed >/dev/termination-log
    exit 0
  fi
}

check_trainer_ret() {
  ret=$1
  stdbuf -oL echo "job returned $ret...setting pod return message..."
  stdbuf -oL echo "==============================="

  if [ $ret -eq 136 ]; then
    echo "Error Arithmetic Operation(Floating Point Exception)" >/dev/termination-log
  elif [ $ret -eq 139 ]; then
    echo "Segmentation Fault" >/dev/termination-log
  elif [ $ret -eq 1 ]; then
    echo "General Error" >/dev/termination-log
  elif [ $ret -eq 134 ]; then
    echo "Program Abort" >/dev/termination-log
  fi
  stdbuf -oL echo "termination log wroted..."
  exit $ret
}

start_fluid_process() {
  pserver_label="paddle-job-pserver=${PADDLE_JOB_NAME}"
  trainer_label="paddle-job=${PADDLE_JOB_NAME}"
  hostname=${HOSTNAME}
  task_index=""

  if [ "${PADDLE_TRAINING_ROLE}" == "TRAINER" ] || [ "${PADDLE_TRAINING_ROLE}" == "PSERVER" ]; then
    stdbuf -oL python -m paddlerec.tools.k8s_tools wait_pods_running ${pserver_label} ${PADDLE_PSERVERS_NUM}
  fi

  export PADDLE_PSERVERS_IP_PORT_LIST=$(python -m paddlerec.tools.k8s_tools fetch_endpoints ${pserver_label} ${PADDLE_PORT})
  export PADDLE_TRAINER_IPS=$(python -m paddlerec.tools.k8s_tools fetch_ips ${trainer_label})

  if [ "${PADDLE_TRAINING_ROLE}" == "TRAINER" ]; then
    check_failed_cnt 1
    task_index=$(python -m paddlerec.tools.k8s_tools fetch_id ${trainer_label})
  else
    task_index=$(python -m paddlerec.tools.k8s_tools fetch_id ${pserver_label})
  fi

  export PADDLE_TRAINER_ID=${task_index}
  export PADDLE_PSERVER_ID=${task_index}

  stdbuf -oL sh -c "${ENTRY}"
  check_trainer_ret $?
}

usage() {
  echo "usage: paddle_k8s [<args>]:"
  echo "  start_fluid Start paddle fluid distributed training, set env"
}

case "$1" in
start_fluid)
  start_fluid_process
  ;;
--help)
  usage
  ;;
*)
  usage
  ;;
esac
