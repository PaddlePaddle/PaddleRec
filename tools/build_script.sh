#!/usr/bin/env bash

# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#=================================================
#                   Utils
#=================================================

set -ex

function init() {
    RED='\033[0;31m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NONE='\033[0m'

    ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../" && pwd )"
}

function check_style() {
  set -e

  export PATH=/usr/bin:$PATH
  pre-commit install

  if ! pre-commit run -a; then
    git diff
    exit 1
  fi

  exit 0
}

function model_test() {
  set -e
  export PATH=/usr/bin:$PATH
  
  root_dir=`pwd`
  all_model=$(find ${root_dir} -name config.yaml)
  special_models=("demo" "pnn" "fgcnn" "gru4rec" "tagspace" "textcnn_pretrain")

  for model in ${all_model}
  do
    skip_flag=false
    for special in ${special_models[*]}
    do
      if [[ $model == *$special* ]]
        then
        echo "Skip "$model
        skip_flag=true
        continue
      fi
    done
    if [[ ${skip_flag} == true ]]
    then
      continue
    fi
    echo "Running "$model
    python -m paddlerec.run -m ${model}

    status=$(echo $?)
    if [[ ${status} -ne 0 ]]
      then
        echo "Test Failed! "$model  
        exit 1  
    fi
    echo "Model Passed "$model
  done
  echo "All Model Test Passed!"
  exit 0
}

function main() {
  local CMD=$1
  init
  case $CMD in
    check_style)
    check_style
    ;;
    model_test)
    model_test
    ;;
    *)
    echo "build failed"
    exit 1
    ;;
    esac
    echo "check_style finished as expected"
}

main $@
