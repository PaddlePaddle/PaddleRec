#!/bin/bash
SOURCE_HOME=$(readlink -f $(dirname ${BASH_SOURCE[0]}) )/
PGLBOX_HOME=${SOURCE_HOME}/../../../
source ${SOURCE_HOME}/pglbox_util.sh
info_log 'Downloading graph data...'

function correct_hadoop_path() {
    # only support "afs:/your/hadoop/path" or "/your/hadoop/path"
    # do not support "afs://xxx.afs.baidu.com:9902/your/hadoop/path" format
    local_path=$1
    if [[ ${local_path} =~ "baidu.com:" ]]; then
        res=`echo ${local_path} | awk -F"/" '{for(i=4;i<=NF;i++){if(i==4){printf("/%s/", $i)}else if(i<NF){printf("%s/", $i)}else{printf("%s", $i)}}}' | awk -F"%" '{print $1}'`
        echo ${res}
    else
        echo ${local_path}
    fi
}

function download_data_by_hadoop() {
    # pushd ${SOURCE_HOME}/../src
    info_log "Downloading data from [$1] to [$2] by hadoop"
    local graph_data_hdfs_path=$1
    local graph_data_local_path=$2
    graph_data_fs_name=`parse_yaml2 ${config_file} graph_data_fs_name`
    graph_data_fs_ugi=`parse_yaml2 ${config_file} graph_data_fs_ugi`
    rm -rf ${graph_data_local_path}
    mkdir -p ${graph_data_local_path}
    python ${SOURCE_HOME}/download_graph_data.py ${graph_data_hdfs_path} \
                                                 ${graph_data_fs_name} \
                                                 ${graph_data_fs_ugi} \
                                                 ${PGLBOX_HOME}/dependency/hadoop-client \
                                                 ${graph_data_local_path}
    if [ $? -ne 0 ]; then
        touch ${local_path}/download.fail
        fatal_log_and_exit "Fail to download_graph_data by python"
    fi
    info_log "Downloaded data from [$1] to [$2] by hadoop"
    # popd
}

function download_data_by_gzshell() {
    # pushd ${SOURCE_HOME}/../
    info_log "Downloading data from [$1] to [$2] by gzshell"
    local graph_data_hdfs_path=$1
    local graph_data_local_path=$2
    graph_data_hdfs_path=`correct_hadoop_path ${graph_data_hdfs_path}`
    graph_data_dir=`echo ${graph_data_hdfs_path} | awk -F'/' '{print $(NF);}'`
    if [ "X${graph_data_dir}" == "X" ]; then
        graph_data_dir=`echo ${graph_data_hdfs_path} | awk -F'/' '{print $(NF-1);}'`
    fi

    fs_name=`parse_yaml2 ${config_file} graph_data_fs_name`
    fs_ugi=`parse_yaml2 ${config_file} graph_data_fs_ugi`
    fs_user=`echo $fs_ugi | awk -F',' '{print $1}' | sed "s/ //g"`
    fs_passwd=`echo $fs_ugi | awk -F',' '{print $2}' | sed "s/ //g"`

    rm -rf ${graph_data_local_path}
    mkdir -p ${graph_data_local_path}
    ${PGLBOX_HOME}/dependency/gzshell --uri=${fs_name} --username=${fs_user} --password=${fs_passwd} --conf=${SOURCE_HOME}/scripts/client.conf --thread=200 -get ${graph_data_hdfs_path} ${graph_data_local_path}
    if [ $? -eq 0 ]; then
        if [ -d ${graph_data_local_path}/${graph_data_dir} ]; then
            mv ${graph_data_local_path}/${graph_data_dir}/* ${graph_data_local_path}/
            rm -rf ${graph_data_local_path}/${graph_data_dir}
        fi
        info_log "Downloaded data from [${hdfs_path}] to [${local_path}]"
    else
        warn_log "Fail to download data from [${hdfs_path}] to [${local_path}] by gzshell, use hadoop-client retry"
        download_data_by_hadoop $1 $2
    fi
    # popd
}

info_log "======================== [BUILD_INFO] download_gdata =============================="

if [ $# = 3 ]; then
    hdfs_path=$1
    local_path=$2
    config_file=$3
else
    hdfs_path=`parse_yaml2 ${config_file} graph_data_hdfs_path`
    local_path=`parse_yaml2 ${config_file} graph_data_local_path`
    config_file="${PGLBOX_HOME}/models/graph/config.yaml" # default config file
fi

rm -rf ${local_path}/download.done
rm -rf ${local_path}/download.fail

fs_name=`parse_yaml2 ${config_file} graph_data_fs_name`
if [[ ${fs_name} =~ "hdfs" ]]; then
    info_log "Downloading graph data by hadoop from [${hdfs_path}] to [${local_path}]"
    download_data_by_hadoop ${hdfs_path} ${local_path}
else
    info_log "Downloading graph data by gzshell from [${hdfs_path}] to [${local_path}]"
    download_data_by_gzshell ${hdfs_path} ${local_path}
fi
if [ $? -ne 0 ]; then
    pushd ${PGLBOX_HOME}/
    touch ${local_path}/download.fail
    popd
    fatal_log_and_exit "download.failed"
fi

pushd ${PGLBOX_HOME}/
info_log "Processing gz files"
sh ${SOURCE_HOME}/graph_process_gz.sh ${local_path}
if [ $? -ne 0 ]; then
    touch ${local_path}/download.fail
    fatal_log_and_exit "process gz file failed"
fi
info_log "Processed gz files"

del_token_padding=`parse_yaml2 ${config_file} del_token_padding`
if [ ${del_token_padding}x = "True"x ] || [ ${del_token_padding}x = "true"x ]; then
   info_log "Deleting zero token"
   ntype2files=`parse_yaml2 ${config_file} ntype2files`
   python ${SOURCE_HOME}/graph_del_zero_token.py ${local_path} ${ntype2files}
   if [ $? -ne 0 ]; then
       touch ${local_path}/download.fail
       fatal_log_and_exit "del zero token failed"
   fi
   info_log "Deleted zero token"
fi

auto_shard=`parse_yaml2 ${config_file} auto_shard`
if [ ${auto_shard}x = "True"x ] || [ ${auto_shard}x = "true"x ]; then
    info_log "Sharding"
    symmetry=`parse_yaml2 ${config_file} symmetry`
    num_part=`parse_yaml2 ${config_file} num_part`
    etype2files=`parse_yaml2 ${config_file} etype2files | awk -F',' '{for(i=1;i<=NF;i++) print $i;}' | awk -F':' '{printf("%s,", $2);}'`
    echo "etype2files: $etype2files"
    ntype2files=`parse_yaml2 ${config_file} ntype2files | awk -F',' '{for(i=1;i<=NF;i++) print $i;}' | awk -F':' '{printf("%s,", $2);}'`
    echo "ntype2files: $ntype2files"
    infer_nodes=`parse_yaml2 ${config_file} infer_nodes`
    echo "infer_nodes: $infer_nodes"
    train_start_nodes=`parse_yaml2 ${config_file} train_start_nodes`
    echo "train_start_nodes: $train_start_nodes"
    type2files="$etype2files$ntype2files,$infer_nodes,$train_start_nodes"
    mock_float_feature=`parse_yaml2 ${config_file} mock_float_feature`
    if [ ${symmetry}x = "True"x ] || [ ${symmetry}x = "true"x ]; then
        if [ ${mock_float_feature}x = "True"x ] || [ ${mock_float_feature}x = "true"x ]; then
            python ${SOURCE_HOME}/graph_sharding.py --input_dir ${local_path} \
                                                    --input_sub_dir ${type2files} \
                                                    --num_part ${num_part} \
                                                    --symmetry \
                                                    --mock_float_feature
        else
            python ${SOURCE_HOME}/graph_sharding.py --input_dir ${local_path} \
                                                    --input_sub_dir ${type2files} \
                                                    --num_part ${num_part} \
                                                    --symmetry
        fi
    else
        if [ ${mock_float_feature}x = "True"x ] || [ ${mock_float_feature}x = "true"x ]; then
            python ${SOURCE_HOME}/graph_sharding.py --input_dir ${local_path} \
                                                    --input_sub_dir ${type2files} \
                                                    --num_part ${num_part} \
                                                    --mock_float_feature
        else
            python ${SOURCE_HOME}/graph_sharding.py --input_dir ${local_path} \
                                                    --input_sub_dir ${type2files} \
                                                    --num_part ${num_part}
        fi
    fi
    info_log "Sharded"
fi

if [ $? -ne 0 ]; then
    touch ${local_path}/download.fail
    fatal_log_and_exit "graph sharding failed"
fi

touch ${local_path}/download.done
popd

info_log 'Downloaded graph data'
