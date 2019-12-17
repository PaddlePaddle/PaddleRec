#!/bin/bash
WORKDIR=`pwd`

echo "current:"$WORKDIR

mpirun -npernode 1 mv package/* ./

export LIBRARY_PATH=$WORKDIR/python/lib:$LIBRARY_PATH
export HADOOP_HOME="${WORKDIR}/hadoop-client/hadoop"

ulimit -c unlimited

mpirun -npernode 1 sh clear_ssd.sh
mpirun -npernode 2 -timestamp-output -tag-output python/bin/python -u trainer_online.py

if [[ $? -ne 0 ]]; then
    echo "Failed to run mpi!" 1>&2
    exit 1
fi
