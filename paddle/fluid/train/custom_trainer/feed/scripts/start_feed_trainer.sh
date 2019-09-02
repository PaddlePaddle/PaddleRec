#!/bin/bash
BIN_FILE=feed_trainer
work_dir=`pwd`

function usage() {
    echo -e "\033[41mUSAGE: sh scripts/start_feed_trainer.sh [run_mode]\033[0m"
    echo "run_mode=mpi, run job in mpi cluster"
    echo "run_mode=mpi_tmp, run 1 node with mpi in /tmp"
    echo "run_mode=local, run 1 node in local"
    echo "Example: sh scripts/start_feed_trainer.sh local"
    exit 0
}
if [ $# -lt 1 ];then
    run_mode="mpi"
else
    run_mode="$1"
fi

export PATH=/usr/local/openmpi/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/openmpi/lib/
if [ "${run_mode}" = "mpi" ];then
    mpirun mv package/* .
    mpirun mkdir -p log
    export HADOOP_HOME="./hadoop-client/hadoop"
    export PATH=$HADOOP_HOME/bin/:./bin:$PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./so
    mpirun sed -i 's/LocalRuntimeEnvironment/MPIRuntimeEnvironment/g' conf/*.yaml
    export HADOOP_HOME="./hadoop-client/hadoop"
    export PATH=$HADOOP_HOME/bin/:/bin:$PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./so

    GLOG_logtostderr=0 mpirun -npernode 2 -timestamp-output -tag-output --prefix $work_dir ./bin/feed_trainer --log_dir=log 
elif [ "${run_mode}" = "mpi_tmp" ];then
    mv package/* .
    mkdir temp
    export HADOOP_HOME="$work_dir/hadoop-client/hadoop"
    export PATH=$HADOOP_HOME/bin/:/bin:$PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${work_dir}/so
    sed -i 's/LocalRuntimeEnvironment/MPIRuntimeEnvironment/g' conf/*.yaml
    mpirun -npernode 2 -timestamp-output -tag-output --prefix $work_dir --mca orte_tmpdir_base ${work_dir}/temp scripts/start_feed_trainer.sh random_log
elif [ "${run_mode}" = "local" ];then
    sed -i 's/MPIRuntimeEnvironment/LocalRuntimeEnvironment/g' conf/*.yaml
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${work_dir}/so
    mkdir log
    ./bin/feed_trainer --log_dir=log
elif [ "${run_mode}" = "random_log" ];then
    log_dir="log/log.${RANDOM}"
    ./bin/feed_trainer --log_dir=log
else
    usage
fi
