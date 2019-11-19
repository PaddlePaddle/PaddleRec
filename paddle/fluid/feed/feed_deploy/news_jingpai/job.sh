#!/bin/bash
WORKDIR=`pwd`

echo "current:"$WORKDIR

mpirun -npernode 1 mv package/* ./

export LIBRARY_PATH=$WORKDIR/python/lib:$LIBRARY_PATH

ulimit -c unlimited

#export FLAGS_check_nan_inf=True
#export check_nan_inf=True

#FLAGS_check_nan_inf=True check_nan_inf=True

#mpirun -npernode 2 -timestamp-output -tag-output -machinefile ${PBS_NODEFILE} python/bin/python -u trainer_online.py

mpirun -npernode 2 -timestamp-output -tag-output python/bin/python -u trainer_online.py

if [[ $? -ne 0 ]]; then
    echo "Failed to run mpi!" 1>&2
    exit 1
fi
