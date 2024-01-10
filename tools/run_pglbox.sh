#!/bin/bash
if [ ! -d "./log" ]; then
  mkdir ./log
  echo "Create log floder for store running log"
fi
# environment variables for fleet distribute training
echo 'begin to train...'

export GLOG_v=0
ulimit -c unlimited

# download dependency
wget https://paddlerec.bj.bcebos.com/benchmark/dependency.tar.gz --no-check-certificate
tar -zxvf dependency.tar.gz
rm dependency.tar.gz

# download data
wget https://paddlerec.bj.bcebos.com/benchmark/pgl/data.tar.gz --no-check-certificate
tar -zxvf data.tar.gz
rm data.tar.gz

# environment variables for fleet distribute training
export FLAGS_dynamic_static_unified_comm=false #PGLBOX不支持新通信库
export NCCL_DEBUG=INFO
export FLAGS_LAUNCH_BARRIER=0
export PADDLE_TRAINERS=1
export FLAGS_enable_tracker_all2all=false
export FLAGS_enable_auto_rdma_trans=true
export FLAGS_enable_all2all_use_fp16=false
export FLAGS_check_nan_inf=false
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_memory_fraction_of_eager_deletion=1
export FLAGS_control_flow_use_new_executor=false
export FLAGS_graph_neighbor_size_percent=1.0
export FLAGS_graph_edges_split_mode="hard"
export FLAGS_enable_graph_multi_node_sampling=false
export FLAGS_new_executor_use_local_scope=0

export PADDLE_TRAINERS_NUM=${PADDLE_TRAINERS}
export PADDLE_PSERVERS_IP_PORT_LIST="127.0.0.1:29011"  #set free port if 29011 is occupied
export PADDLE_TRAINER_ENDPOINTS=${PADDLE_TRAINER_ENDPOINTS/,*/}
export PADDLE_PSERVER_PORT_ARRAY=(29011)
export PADDLE_TRAINER_ID=0
export TRAINING_ROLE=TRAINER
export PADDLE_PORT=8800

#PYTHON_HOME=./dependency/cpython-3.7.0
#export PATH=${PYTHON_HOME}/bin:${PATH}
#export LD_LIBRARY_PATH=${PYTHON_HOME}/lib:${LD_LIBRARY_PATH}
export LD_PRELOAD=./dependency/libjemalloc.so
export DEPENDENCY_HOME=./dependency
export GZSHELL=./dependency/gzshell

export FLAGS_free_when_no_cache_hit=true
export FLAGS_use_stream_safe_cuda_allocator=true
export FLAGS_gpugraph_enable_hbm_table_collision_stat=false
export FLAGS_gpugraph_hbm_table_load_factor=0.75
export FLAGS_gpugraph_enable_segment_merge_grads=true
export FLAGS_gpugraph_merge_grads_segment_size=128
export FLAGS_gpugraph_dedup_pull_push_mode=1
export FLAGS_gpugraph_load_node_list_into_hbm=false
# storage mode
# 1. WHOLE_HBM 2.MEM_EMBEDDING_NO_FEATURE"(currently not supported)
# 3.MEM_EMBEDDING 4.SSD_EMBEDDING
train_storage_mode=`grep train_storage_mode $config_file | sed s/#.*//g | grep train_storage_mode | awk -F':' '{print $2}' | sed 's/ //g'`
if [ "${train_storage_mode}" = "WHOLE_HBM" ]; then
    export FLAGS_gpugraph_storage_mode=1
    echo "FLAGS_gpugraph_storage_mode is WHOLE_HBM"
elif [ "${train_storage_mode}" = "SSD_EMBEDDING" ]; then
    export FLAGS_gpugraph_storage_mode=4
    echo "FLAGS_gpugraph_storage_mode is SSD_EMBEDDING"
else
    export FLAGS_gpugraph_storage_mode=3
    echo "FLAGS_gpugraph_storage_mode is MEM_EMBEDDING"
fi


export FLAGS_enable_exit_when_partial_worker=false
export FLAGS_gpugraph_debug_gpu_memory=false
export FLAGS_gpugraph_slot_feasign_max_num=200
    

export FLAGS_graph_load_in_parallel=true

    
export FLAGS_graph_get_neighbor_id=true

export FLAGS_graph_metapath_split_opt=false
    
export FLAGS_rocksdb_path="./database"


    
#python3.7 -c 'import paddle; print(paddle.version.commit)';
set -x
which python
unset PYTHONHOME
unset PYTHONPATH

ret=0
for((i=0;i<$PADDLE_TRAINERS;i++))
do
    python -u tools/static_pglbox_trainer.py -m models/graph/lightgcn.yaml &> ./log/tainer.$i.log

done
ret=$?

if [[ $ret -ne 0 ]]; then
    echo "Something failed in cluster_train_and_infer.py"
    exit 1
fi
