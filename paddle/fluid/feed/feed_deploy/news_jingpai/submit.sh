#!/bin/bash

source ./package/my_nets/config.py

rm -r tmp/*
mkdir tmp
cd tmp

mkdir ./package
cp -r ../package/python ./package
cp -r ../package/my_nets/* ./package
cp ../qsub_f.conf ./
cp ../job.sh ./
cp ../job.sh ./package

if [ "a${sparse_table_storage}" = "assd" ];then
    sed -i 's/DownpourSparseTable/DownpourSparseSSDTable' ./package/my_nets/reqi_fleet_desc
fi

current=`date "+%Y-%m-%d %H:%M:%S"`
timeStamp=`date -d "$current" +%s`
output_path=${output_path#*:}
hdfs_output=${output_path}/$timeStamp

export HADOOP_HOME="${local_hadoop_home}"

MPI_NODE_MEM=${node_memory}
echo "SERVER=${mpi_server}" > qsub_f.conf
echo "QUEUE=${mpi_queue}" >> qsub_f.conf
echo "PRIORITY=${mpi_priority}" >> qsub_f.conf
echo "USE_FLAGS_ADVRES=yes" >> qsub_f.conf

if [ "a${sparse_table_storage}" = "assd" ];then
    ${smart_client_home}/bin/qsub_f \
    -N $task_name \
    --conf ./qsub_f.conf \
    --hdfs $fs_name  \
    --ugi $fs_ugi \
    --hout $hdfs_output \
    --am-type smart_am \
    --files ./package \
    --workspace /home/ssd1/normandy/maybach \
    -l nodes=$nodes,walltime=1000:00:00,pmem-hard=$MPI_NODE_MEM,pcpu-soft=280,pnetin-soft=1000,pnetout-soft=1000 ./job.sh
else
    ${smart_client_home}/bin/qsub_f \
    -N $task_name \
    --conf ./qsub_f.conf \
    --hdfs $fs_name  \
    --ugi $fs_ugi \
    --hout $hdfs_output \
    --am-type smart_am \
    --files ./package \
    -l nodes=$nodes,walltime=1000:00:00,pmem-hard=$MPI_NODE_MEM,pcpu-soft=280,pnetin-soft=1000,pnetout-soft=1000 ./job.sh
fi
