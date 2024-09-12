#!/bin/bash
# environment variables for fleet distribute training
echo 'begin to train...'

export GLOG_v=0
ulimit -c unlimited

# download dependency
wget https://paddlerec.bj.bcebos.com/benchmark/pgl/dependency_py310.tar.gz --no-check-certificate
tar -zxvf dependency_py310.tar.gz
rm dependency_py310.tar.gz

SOURCE_HOME=$(readlink -f $(dirname ${BASH_SOURCE[0]}) )/
PGLBOX_HOME=${SOURCE_HOME}/../
LOG_DIR="${PGLBOX_HOME}/log"
[ ! -d ${LOG_DIR} ] && mkdir -p ${LOG_DIR}

config_file="${PGLBOX_HOME}/models/graph/config.yaml" # 模型配置文件，可以修改

# environment variables for fleet distribute training
source ${PGLBOX_HOME}/tools/utils/static_ps/pglbox_util.sh

unset PYTHONHOME
unset PYTHONPATH

# download graph data
graph_data_hdfs_path=`parse_yaml2 ${config_file} graph_data_hdfs_path`
graph_data_local_path=`parse_yaml2 ${config_file} graph_data_local_path`
if [ -z "$graph_data_hdfs_path" ]; then
    echo "download default graph data"
    wget https://paddlerec.bj.bcebos.com/benchmark/pgl/data.tar.gz --no-check-certificate
    tar -zxvf data.tar.gz
    rm data.tar.gz
    touch ${PGLBOX_HOME}/data/download.done
else
    echo "download your graph data"
    sh ${PGLBOX_HOME}/tools/utils/static_ps/download_graph_data.sh ${graph_data_hdfs_path} ${graph_data_local_path} ${config_file}> ${LOG_DIR}/graph_data.log 2>&1 &
fi

# train
sharding=`grep sharding $config_file | sed s/#.*//g | grep sharding | awk -F':' '{print $1}' | sed 's/ //g'`
if [ "${sharding}" = "sharding" ]; then
   export FLAGS_enable_adjust_op_order=2
fi

pretrained_model=`parse_yaml2 $config_file pretrained_model`
sage_mode=`parse_yaml2 $config_file sage_mode`
if [[ ${pretrained_model} =~ "1.5B" ]] || [[ ${pretrained_model} =~ "10B" ]]; then
    echo "pretrained_model is [${pretrained_model}], using LLM_MODELING"
    export LLM_MODELING=true
fi

# environment variables for fleet distribute training
export FLAGS_enable_pir_api=0 #PS模式不支持新IR
export FLAGS_dynamic_static_unified_comm=false #PGLBOX最新不支持新通信库
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

# multiple machines need high version nccl
# set launch mode GROUP, after more than nccl2.9 default PARALLEL multi-thread will blocking
export NCCL_LAUNCH_MODE=GROUP
# export NCCL_ROOT=/home/work/nccl/nccl2.16.2_cuda11  # 注意nccl位置
# export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$LD_LIBRARY_PATH

if [[ ! -z "$MPI_NODE_NUM" ]] && [[ $MPI_NODE_NUM -gt 1 ]]; then
    echo "PADDLE_TRAINER_ID: $PADDLE_TRAINER_ID, PADDLE_TRAINER_ENDPOINTS: $PADDLE_TRAINER_ENDPOINTS, PADDLE_CURRENT_ENDPOINT: $PADDLE_CURRENT_ENDPOINT"
    export PADDLE_WITH_GLOO=2
    export PADDLE_GLOO_RENDEZVOUS=3
    export FLAGS_graph_edges_split_only_by_src_id=true
    export FLAGS_enable_graph_multi_node_sampling=true
    if [ $FLAGS_graph_edges_split_mode = "hard" ]; then
        echo "run gpugraph in hard mode"
    elif [ $FLAGS_graph_edges_split_mode = "fennel" ]; then
        export FLAGS_enable_sparse_inner_gather=false
        export FLAGS_query_dest_rank_by_multi_node=true
        echo "run gpugraph in fennel mode"
    fi

    if [ $sage_mode = "True" ]; then
        export FLAGS_graph_embedding_split_infer_mode=false
        echo "run gpugraph in sage mode"
    else
        export FLAGS_graph_embedding_split_infer_mode=true
        echo "run gpugraph in deepwalk mode"
    fi
else
    export PADDLE_TRAINERS_NUM=${PADDLE_TRAINERS}
    export POD_IP=127.0.0.1
    export PADDLE_PSERVERS_IP_PORT_LIST="127.0.0.1:29011"  #set free port if 29011 is occupied
    export PADDLE_TRAINER_ENDPOINTS=${PADDLE_TRAINER_ENDPOINTS/,*/}
    export PADDLE_PSERVER_PORT_ARRAY=(29011)
    export PADDLE_TRAINER_ID=0
    export TRAINING_ROLE=TRAINER
    export PADDLE_PORT=8800
fi

export LD_PRELOAD=./dependency/libjemalloc.so
# jemalloc parameter tuning
export MALLOC_CONF=background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000
export DEPENDENCY_HOME=./dependency
export HADOOP_HOME="${PGLBOX_HOME}/dependency/hadoop-client/hadoop"
export PATH=${PGLBOX_HOME}/dependency/hadoop-client/hadoop/bin:$PATH
export GZSHELL="${PGLBOX_HOME}/dependency/gzshell"

selected_gpus=`grep gpus: $config_file |sed s/#.*//g | grep gpus | awk -F':' '{print $2}' | sed "s/\[//g" | sed "s/\]//g" | sed "s/ //g"`
export FLAGS_selected_gpus=${selected_gpus}
export FLAGS_free_when_no_cache_hit=true
export FLAGS_use_stream_safe_cuda_allocator=true
export FLAGS_gpugraph_enable_hbm_table_collision_stat=false
export FLAGS_gpugraph_hbm_table_load_factor=0.75
export FLAGS_gpugraph_enable_segment_merge_grads=true
export FLAGS_gpugraph_merge_grads_segment_size=128
export FLAGS_gpugraph_dedup_pull_push_mode=1
export FLAGS_gpugraph_load_node_list_into_hbm=false

export FLAGS_gpugraph_storage_mode=3

export FLAGS_enable_exit_when_partial_worker=false
export FLAGS_gpugraph_debug_gpu_memory=false
export FLAGS_enable_neighbor_list_use_uva=false
max_seq_len=`parse_yaml2 $config_file max_seq_len`
if [ "$max_seq_len" != "" ]; then
    export FLAGS_gpugraph_slot_feasign_max_num=${max_seq_len}
else
    export FLAGS_gpugraph_slot_feasign_max_num=200
fi
    
export FLAGS_gpugraph_enable_gpu_direct_access=false
export FLAGS_graph_load_in_parallel=true
sage_mode=`grep sage_mode $config_file | sed s/#.*//g | grep sage_mode | awk -F':' '{print $2}' | sed 's/ //g'`
if [ "${sage_mode}" = "True" ] || [ "${sage_mode}" = "true" ]; then
    export FLAGS_enable_exit_when_partial_worker=true
    echo "FLAGS_enable_exit_when_partial_worker is true"
else
    export FLAGS_enable_exit_when_partial_worker=false
    echo "FLAGS_enable_exit_when_partial_worker is false"
fi

metapath_split_opt=`grep metapath_split_opt $config_file | sed s/#.*//g | grep metapath_split_opt | awk -F':' '{print $2}' | sed 's/ //g'`
if [ "${metapath_split_opt}" == "True" ] || [ "${metapath_split_opt}" == "true" ];then
    export FLAGS_graph_metapath_split_opt=true
    echo "FLAGS_graph_metapath_split_opt is true"
else
    export FLAGS_graph_metapath_split_opt=false
    echo "FLAGS_graph_metapath_split_opt is false"
fi
    
part_num=`grep num_part $config_file | sed s/#.*//g | grep num_part | awk -F':' '{print $2}' | sed 's/ //g'`
if [ ${part_num} -eq 1000 ];then
    echo "will run full graph"
    export FLAGS_graph_get_neighbor_id=false
else
    echo "will sub part graph"
    export FLAGS_graph_get_neighbor_id=true
fi

data_path=`parse_yaml2 $config_file graph_data_local_path`
echo "data_path:"$data_path
if [[ ${data_path} =~ "raid0" ]]; then
    echo "set export FLAGS_rocksdb_path=/raid0/database"
    export FLAGS_rocksdb_path="/raid0/database"
fi

PYTHON_HOME=${PGLBOX_HOME}/dependency/cpython-3.10.0
export PATH=${PYTHON_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${PYTHON_HOME}/lib:${LD_LIBRARY_PATH}
set -x
which python
unset PYTHONHOME
unset PYTHONPATH

# install paddlepaddle-gpu whl
# python -m pip install paddlepaddle-gpu==2.6.1

ret=0
for((i=0;i<$PADDLE_TRAINERS;i++))
do
    python -u tools/static_pglbox_trainer.py -m $config_file &> ./log/trainer.$i.log

done
ret=$?

if [[ $ret -ne 0 ]]; then
    echo "Something failed in cluster_train_and_infer.py"
    exit 1
fi
