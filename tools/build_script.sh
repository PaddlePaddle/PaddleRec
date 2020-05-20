function init() {
    RED='\033[0;31m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NONE='\033[0m'

    ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../" && pwd )"
}

function check_style() {
  trap 'abort' 0
  set -e

  if [ -x "$(command -v gimme)" ]; then
    eval "$(GIMME_GO_VERSION=1.8.3 gimme)"
  fi

  pip install cpplint
  # set up go environment for running gometalinter
  mkdir -p $GOPATH/src/github.com/SeiriosPlus/PaddleRec
  ln -sf ${ROOT} $GOPATH/src/github.com/SeiriosPlus/PaddleRec

  export PATH=/usr/bin:$PATH
  pre-commit install
  clang-format --version

  if ! pre-commit run -a; then
    git diff
    exit 1
  fi

  trap : 0
}

function main() {
  local CMD=$1
  init
  case $CMD in
    check_style)
    check_style
    ;;
}

main $@
