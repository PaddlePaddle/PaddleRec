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

function main() {
  local CMD=$1
  init
  case $CMD in
    check_style)
    check_style
    ;;
      *)
    echo "build failed"
    exit 1
        ;;
    esac
    echo "check_style finished as expected"
}

main $@
