#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1
# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer', 'whole_infer', 'klquant_whole_infer']
MODE=$2
if [ ${MODE} = "benchmark_train" ]; then
    dataline=$(awk 'NR==1, NR==60{print}'  $FILENAME)
else
    dataline=$(awk 'NR==1, NR==51{print}'  $FILENAME)
fi

# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")
python=$(func_parser_value "${lines[2]}")
gpu_list=$(func_parser_value "${lines[3]}")
train_use_gpu_key=$(func_parser_key "${lines[4]}")
train_use_gpu_value=$(func_parser_value "${lines[4]}")
autocast_list=$(func_parser_value "${lines[5]}")
autocast_key=$(func_parser_key "${lines[5]}")
epoch_key=$(func_parser_key "${lines[6]}")
epoch_num=$(func_parser_params "${lines[6]}" "${MODE}")
save_model_key=$(func_parser_key "${lines[7]}")
train_batch_key=$(func_parser_key "${lines[8]}")
train_batch_value=$(func_parser_params "${lines[8]}" "${MODE}")
pretrain_model_key=$(func_parser_key "${lines[9]}")
pretrain_model_value=$(func_parser_value "${lines[9]}")
train_model_name=$(func_parser_params "${lines[10]}" "${MODE}")
train_infer_img_dir=$(func_parser_value "${lines[11]}")
train_param_key1=$(func_parser_key "${lines[12]}")
train_param_value1=$(func_parser_value "${lines[12]}")

trainer_list=$(func_parser_value "${lines[14]}")
trainer_norm=$(func_parser_key "${lines[15]}")
norm_trainer=$(func_parser_value "${lines[15]}")
pact_key=$(func_parser_key "${lines[16]}")
pact_trainer=$(func_parser_value "${lines[16]}")
fpgm_key=$(func_parser_key "${lines[17]}")
fpgm_trainer=$(func_parser_value "${lines[17]}")
distill_key=$(func_parser_key "${lines[18]}")
distill_trainer=$(func_parser_value "${lines[18]}")
trainer_key1=$(func_parser_key "${lines[19]}")
trainer_value1=$(func_parser_value "${lines[19]}")
trainer_key2=$(func_parser_key "${lines[20]}")
trainer_value2=$(func_parser_value "${lines[20]}")

eval_py=$(func_parser_value "${lines[23]}")
eval_key1=$(func_parser_key "${lines[24]}")
eval_value1=$(func_parser_value "${lines[24]}")

save_infer_key=$(func_parser_key "${lines[27]}")
export_weight=$(func_parser_key "${lines[28]}")
norm_export=$(func_parser_value "${lines[29]}")
pact_export=$(func_parser_value "${lines[30]}")
fpgm_export=$(func_parser_value "${lines[31]}")
distill_export=$(func_parser_value "${lines[32]}")
export_key1=$(func_parser_key "${lines[33]}")
export_value1=$(func_parser_value "${lines[33]}")
export_key2=$(func_parser_key "${lines[34]}")
export_value2=$(func_parser_value "${lines[34]}")
inference_dir=$(func_parser_value "${lines[35]}")

# parser inference model 
infer_model_dir_list=$(func_parser_value "${lines[36]}")
infer_export_list=$(func_parser_value "${lines[37]}")
infer_is_quant=$(func_parser_value "${lines[38]}")
# parser inference 
inference_py=$(func_parser_value "${lines[39]}")
use_gpu_key=$(func_parser_key "${lines[40]}")
use_gpu_list=$(func_parser_value "${lines[40]}")
use_mkldnn_key=$(func_parser_key "${lines[41]}")
use_mkldnn_list=$(func_parser_value "${lines[41]}")
cpu_threads_key=$(func_parser_key "${lines[42]}")
cpu_threads_list=$(func_parser_value "${lines[42]}")
batch_size_key=$(func_parser_key "${lines[43]}")
batch_size_list=$(func_parser_value "${lines[43]}")
use_trt_key=$(func_parser_key "${lines[44]}")
use_trt_list=$(func_parser_value "${lines[44]}")
precision_key=$(func_parser_key "${lines[45]}")
precision_list=$(func_parser_value "${lines[45]}")
infer_model_key=$(func_parser_key "${lines[46]}")
image_dir_key=$(func_parser_key "${lines[47]}")
infer_img_dir=$(func_parser_value "${lines[47]}")
save_log_key=$(func_parser_key "${lines[48]}")
benchmark_key=$(func_parser_key "${lines[49]}")
benchmark_value=$(func_parser_value "${lines[49]}")
infer_key1=$(func_parser_key "${lines[50]}")
infer_value1=$(func_parser_value "${lines[50]}")

#parser benchmark 
if [ ${MODE} = "benchmark_train" ]; then
    run_mode_key=$(func_parser_key "${lines[55]}")
    run_mode_value=$(func_parser_value "${lines[55]}")
    gpu_config_key=$(func_parser_key "${lines[58]}")
    gpu_config_value=$(func_parser_value "${lines[58]}")
    cpu_config_key=$(func_parser_key "${lines[59]}")
    cpu_config_value=$(func_parser_value "${lines[59]}")
fi


# parser klquant_infer
if [ ${MODE} = "klquant_whole_infer" ]; then
    dataline=$(awk 'NR==1 NR==17{print}'  $FILENAME)
    lines=(${dataline})
    model_name=$(func_parser_value "${lines[1]}")
    python=$(func_parser_value "${lines[2]}")
    # parser inference model 
    infer_model_dir_list=$(func_parser_value "${lines[3]}")
    infer_export_list=$(func_parser_value "${lines[4]}")
    infer_is_quant=$(func_parser_value "${lines[5]}")
    # parser inference 
    inference_py=$(func_parser_value "${lines[6]}")
    use_gpu_key=$(func_parser_key "${lines[7]}")
    use_gpu_list=$(func_parser_value "${lines[7]}")
    use_mkldnn_key=$(func_parser_key "${lines[8]}")
    use_mkldnn_list=$(func_parser_value "${lines[8]}")
    cpu_threads_key=$(func_parser_key "${lines[9]}")
    cpu_threads_list=$(func_parser_value "${lines[9]}")
    batch_size_key=$(func_parser_key "${lines[10]}")
    batch_size_list=$(func_parser_value "${lines[10]}")
    use_trt_key=$(func_parser_key "${lines[11]}")
    use_trt_list=$(func_parser_value "${lines[11]}")
    precision_key=$(func_parser_key "${lines[12]}")
    precision_list=$(func_parser_value "${lines[12]}")
    infer_model_key=$(func_parser_key "${lines[13]}")
    image_dir_key=$(func_parser_key "${lines[14]}")
    infer_img_dir=$(func_parser_value "${lines[14]}")
    save_log_key=$(func_parser_key "${lines[15]}")
    benchmark_key=$(func_parser_key "${lines[16]}")
    benchmark_value=$(func_parser_value "${lines[16]}")
    infer_key1=$(func_parser_key "${lines[17]}")
    infer_value1=$(func_parser_value "${lines[17]}")
fi

LOG_PATH="./test_tipc/output/${model_name}/${MODE}"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results.log"


function func_inference(){
    IFS='|'
    _python=$1
    _script=$2
    _model_dir=$3
    _log_path=$4
    _img_dir=$5
    _flag_quant=$6
    # inference 
    for use_gpu in ${use_gpu_list[*]}; do
        if [ ${use_gpu} = "False" ] || [ ${use_gpu} = "cpu" ]; then
            for use_mkldnn in ${use_mkldnn_list[*]}; do
                if [ ${use_mkldnn} = "False" ] && [ ${_flag_quant} = "True" ]; then
                    continue
                fi
                for threads in ${cpu_threads_list[*]}; do
                    for batch_size in ${batch_size_list[*]}; do
                        for precision in ${precision_list[*]}; do
                            if [ ${use_mkldnn} = "False" ] && [ ${precision} = "fp16" ]; then
                                continue
                            fi # skip when enable fp16 but disable mkldnn
                            if [ ${_flag_quant} = "True" ] && [ ${precision} != "int8" ]; then
                                continue
                            fi # skip when quant model inference but precision is not int8
                            set_precision=$(func_set_params "${precision_key}" "${precision}")
                            
                            _save_log_path="${_log_path}/python_infer_cpu_gpus_${use_gpu}_usemkldnn_${use_mkldnn}_threads_${threads}_precision_${precision}_batchsize_${batch_size}.log"
                            set_infer_data=$(func_set_params "${image_dir_key}" "${_img_dir}")
                            set_benchmark=$(func_set_params "${benchmark_key}" "${benchmark_value}")
                            set_batchsize=$(func_set_params "${batch_size_key}" "${batch_size}")
                            set_cpu_threads=$(func_set_params "${cpu_threads_key}" "${threads}")
                            set_model_dir=$(func_set_params "${infer_model_key}" "${_model_dir}")
                            set_infer_params1=$(func_set_params "${infer_key1}" "${infer_value1}")
                            command="${_python} ${_script} ${use_gpu_key}=${use_gpu} ${use_mkldnn_key}=${use_mkldnn} ${set_cpu_threads} ${set_model_dir} ${set_batchsize} ${set_infer_data} ${set_benchmark} ${set_precision} ${set_infer_params1} > ${_save_log_path} 2>&1 "
                            eval $command
                            last_status=${PIPESTATUS[0]}
                            eval "cat ${_save_log_path}"
                            status_check $last_status "${command}" "${status_log}" "${model_name}" "${_save_log_path}"
                        done
                    done
                done
            done
        elif [ ${use_gpu} = "True" ] || [ ${use_gpu} = "gpu" ]; then
            for use_trt in ${use_trt_list[*]}; do
                for precision in ${precision_list[*]}; do
                    if [[ ${_flag_quant} = "False" ]] && [[ ${precision} =~ "int8" ]]; then
                        continue
                    fi 
                    if [[ ${precision} =~ "fp16" || ${precision} =~ "int8" ]] && [ ${use_trt} = "False" ]; then
                        continue
                    fi
                    if [[ ${use_trt} = "False" || ${precision} =~ "int8" ]] && [ ${_flag_quant} = "True" ]; then
                        continue
                    fi
                    for batch_size in ${batch_size_list[*]}; do
                        _save_log_path="${_log_path}/python_infer_gpu_gpus_${use_gpu}_usetrt_${use_trt}_precision_${precision}_batchsize_${batch_size}.log"
                        set_infer_data=$(func_set_params "${image_dir_key}" "${_img_dir}")
                        set_benchmark=$(func_set_params "${benchmark_key}" "${benchmark_value}")
                        set_batchsize=$(func_set_params "${batch_size_key}" "${batch_size}")
                        set_tensorrt=$(func_set_params "${use_trt_key}" "${use_trt}")
                        set_precision=$(func_set_params "${precision_key}" "${precision}")
                        set_model_dir=$(func_set_params "${infer_model_key}" "${_model_dir}")
                        set_infer_params1=$(func_set_params "${infer_key1}" "${infer_value1}")
                        command="${_python} ${_script} ${use_gpu_key}=${use_gpu} ${set_tensorrt} ${set_precision} ${set_model_dir} ${set_batchsize} ${set_infer_data} ${set_benchmark} ${set_infer_params1} > ${_save_log_path} 2>&1 "
                        eval $command
                        last_status=${PIPESTATUS[0]}
                        eval "cat ${_save_log_path}"
                        status_check $last_status "${command}" "${status_log}" "${model_name}" "${_save_log_path}"
                        
                    done
                done
            done
        else
            echo "Does not support hardware other than CPU and GPU Currently!"
        fi
    done
}

if [ ${MODE} = "benchmark_train" ]; then
	if [ ! -d "./log" ]; then
	  mkdir ./log
	  echo "Create log floder for store running log"
	fi
	if [ ${run_mode_value} = "PSGPU" ]; then
        export FLAGS_dynamic_static_unified_comm=False #PSGPU不支持新通信库
        export FLAGS_LAUNCH_BARRIER=0
        export PADDLE_TRAINER_ID=0
        export PADDLE_PSERVER_NUMS=1
        export PADDLE_TRAINERS=1
        export PADDLE_TRAINERS_NUM=${PADDLE_TRAINERS}
        export POD_IP=127.0.0.1

        # set free port if 29011 is occupied
        export PADDLE_PSERVERS_IP_PORT_LIST="127.0.0.1:29011"
        export PADDLE_PSERVER_PORT_ARRAY=(29011)

        # set gpu numbers according to your device
        #export FLAGS_selected_gpus="0,1,2,3,4,5,6,7"
        export FLAGS_selected_gpus=${gpu_list}

        # set your model yaml
        CONFIG=$gpu_config_value
        SC="tools/static_gpubox_trainer.py -m "
        BATCH="-o runner.train_batch_size="$train_batch_value
        EPOCH="runner.epochs="$epoch_num
        # run pserver
        export TRAINING_ROLE=PSERVER
        for((i=0;i<$PADDLE_PSERVER_NUMS;i++))
        do
            cur_port=${PADDLE_PSERVER_PORT_ARRAY[$i]}
            echo "PADDLE WILL START PSERVER "$cur_port
            export PADDLE_PORT=${cur_port}
            cmd="${python} ${SC} ${CONFIG} ${BATCH} ${EPOCH}"
            eval $cmd
        done

        # run trainer
        export TRAINING_ROLE=TRAINER
        for((i=0;i<$PADDLE_TRAINERS;i++))
        do
            echo "PADDLE WILL START Trainer "$i
            export PADDLE_TRAINER_ID=$i
            cmd="${python} ${SC} ${CONFIG} ${BATCH} ${EPOCH}"
            eval $cmd
        done
    elif [ ${run_mode_value} = "PSCPU" ]; then
        GLOO_PATH="../tools/paddlecloud/config.ini"
        gloo_dataline=$(awk '{print}'  $GLOO_PATH)
        gloo_lines=(${gloo_dataline})
        gloo_fs_name=$(func_parser_gloo_value "${gloo_lines[4]}")
        gloo_fs_ugi=$(func_parser_gloo_value "${gloo_lines[5]}")
        gloo_fs_path=$(func_parser_gloo_value "${gloo_lines[6]}")
        wget https://paddlerec.bj.bcebos.com/benchmark/brpc.tar.gz --no-check-certificate
        tar -zxvf brpc.tar.gz
        BRPC_PATH=`pwd`/brpc
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${BRPC_PATH}
        export PADDLE_WITH_GLOO=1
        export PADDLE_GLOO_RENDEZVOUS=1
        export PADDLE_GLOO_FS_NAME=$gloo_fs_name
        export PADDLE_GLOO_FS_UGI=$gloo_fs_ugi
        export PADDLE_GLOO_FS_PATH=$gloo_fs_path

        cd ../
        ${python} -m pip install --force-reinstall paddlepaddle*.whl
        cd -
        CONFIG=$cpu_config_value
        SC="tools/static_ps_trainer.py -m "
        BATCH="-o runner.train_batch_size="$train_batch_value
        EPOCH="runner.epochs="$epoch_num
        cmd="${python} ${SC} ${CONFIG} ${BATCH} ${EPOCH}"
        eval $cmd

    else
        SOURCE_HOME=$(readlink -f $(dirname ${BASH_SOURCE[0]}) )/
        PGLBOX_HOME=${SOURCE_HOME}/../
        echo SOURCE_HOME:$SOURCE_HOME
        echo PGLBOX_HOME:$PGLBOX_HOME
        #通过sed切换执行配置文件路径
        config_file_line="config_file=${PGLBOX_HOME}/$gpu_config_value"
        sed -i "/config_file=/a\\$config_file_line" tools/run_pglbox.sh 
        #删除设置的重定向的脚本
        line=$(sed -n -e '/static_pglbox_trainer.py/=' tools/run_pglbox.sh)
        sed -i "$line d" tools/run_pglbox.sh
        cmd="python -u tools/static_pglbox_trainer.py -m ${PGLBOX_HOME}/$gpu_config_value"
        sed -i "$line a$cmd" tools/run_pglbox.sh
        #ubuntu环境下有这个文件，centos下没有
        cat /etc/lsb-release
        if [ $? -eq 0 ];then
            #使用PDC的ubuntu镜像需要将环境变量注释掉
            sed -i '/PYTHON_HOME=${PGLBOX_HOME}/d' tools/run_pglbox.sh 
            sed -i '/export PATH=${PYTHON_HOME}/d' tools/run_pglbox.sh 
            sed -i '/export LD_LIBRARY_PATH=${PYTHON_HOME}/d' tools/run_pglbox.sh
        else
            #使用PDC的centos镜像需要提前制定python环境
            sed -i '/tar -zxvf dependency_py310.tar.gz/d' tools/run_pglbox.sh 
            sed -i '/rm dependency_py310.tar.gz/d' tools/run_pglbox.sh 
        fi
        if [[ ${SYS_JOB_NAME} && ${SYS_JOB_NAME} =~ 'CE' ]]; then
            line=$(sed -n -e '/graph_data_fs_name:/=' $gpu_config_value)
            new_graph_data_fs_name="graph_data_fs_name: \"${graph_data_fs_name}\""
            sed -i "$line a${new_graph_data_fs_name}" $gpu_config_value
            sed -i "$line d" $gpu_config_value

            line=$(sed -n -e '/graph_data_fs_ugi:/=' $gpu_config_value)
            new_graph_data_fs_ugi="graph_data_fs_ugi: \"${graph_data_fs_ugi}\""
            sed -i "$line a${new_graph_data_fs_ugi}" $gpu_config_value
            sed -i "$line d" $gpu_config_value

            lines=$(sed -n -e '/graph_data_hdfs_path:/=' $gpu_config_value)
            array_lines=(${lines})
            line_num=${#array_lines[@]}
            line=${array_lines[line_num-1]}
            new_graph_data_hdfs_path="graph_data_hdfs_path: \"${graph_data_hdfs_path}\""
            sed -i "$line a${new_graph_data_hdfs_path}" $gpu_config_value
            sed -i "$line d" $gpu_config_value

            lines=$(sed -n -e '/graph_data_local_path:/=' $gpu_config_value)
            array_lines=(${lines})
            line_num=${#array_lines[@]}
            line=${array_lines[line_num-1]}
            new_graph_data_local_path="graph_data_local_path: \"${graph_data_local_path}\""
            sed -i "$line a${new_graph_data_local_path}" $gpu_config_value
            sed -i "$line d" $gpu_config_value

            lines=$(sed -n -e '/num_part:/=' $gpu_config_value)
            array_lines=(${lines})
            line_num=${#array_lines[@]}
            line=${array_lines[line_num-1]}
            new_num_part="num_part: 1000"
            sed -i "$line a${new_num_part}" $gpu_config_value
            sed -i "$line d" $gpu_config_value

            wget ${graph_eval_url} --no-check-certificate -P tools/
        fi
        #执行训练脚本
        sh -x tools/run_pglbox.sh
        if [[ ${SYS_JOB_NAME} && ${SYS_JOB_NAME} =~ 'CE' ]]; then
            sh tools/run_graph_eval.sh $gpu_config_value > ${BENCHMARK_LOG_DIR}/graph_eval.log 2>&1
            rm -rf ${graph_data_local_path}
        fi
    fi
elif [ ${MODE} = "whole_infer" ] || [ ${MODE} = "klquant_whole_infer" ]; then
    GPUID=$3
    if [ ${#GPUID} -le 0 ];then
        env=" "
    else
        env="export CUDA_VISIBLE_DEVICES=${GPUID}"
    fi
    # set CUDA_VISIBLE_DEVICES
    eval $env
    export Count=0
    IFS="|"
    infer_run_exports=(${infer_export_list})
    infer_quant_flag=(${infer_is_quant})
    for infer_model in ${infer_model_dir_list[*]}; do
        # run export
        if [ ${infer_run_exports[Count]} != "null" ];then
            save_infer_dir=$(dirname $infer_model)
            set_export_weight=$(func_set_params "${export_weight}" "${infer_model}")
            set_save_infer_key=$(func_set_params "${save_infer_key}" "${save_infer_dir}")
            export_cmd="${python} ${infer_run_exports[Count]} ${set_export_weight} ${set_save_infer_key}"
            echo ${infer_run_exports[Count]} 
            echo  $export_cmd
            eval $export_cmd
            status_export=$?
            status_check $status_export "${export_cmd}" "${status_log}"
        else
            save_infer_dir=${infer_model}
        fi
        #run inference
        is_quant=${infer_quant_flag[Count]}
        if [ ${MODE} = "klquant_infer" ]; then
            is_quant="True"
        fi
        func_inference "${python}" "${inference_py}" "${save_infer_dir}" "${LOG_PATH}" "${infer_img_dir}" ${is_quant}
        Count=$(($Count + 1))
    done
else
    IFS="|"
    export Count=0
    USE_GPU_KEY=(${train_use_gpu_value})
    for gpu in ${gpu_list[*]}; do
        train_use_gpu=${USE_GPU_KEY[Count]}
        Count=$(($Count + 1))
        ips=""
        if [ ${gpu} = "-1" ];then
            env=""
        elif [ ${#gpu} -le 1 ];then
            env="export CUDA_VISIBLE_DEVICES=${gpu}"
            eval ${env}
        elif [ ${#gpu} -le 15 ];then
            IFS=","
            array=(${gpu})
            env="export CUDA_VISIBLE_DEVICES=${array[0]}"
            IFS="|"
        else
            IFS=";"
            array=(${gpu})
            ips=${array[0]}
            gpu=${array[1]}
            IFS="|"
            env=" "
        fi
        for autocast in ${autocast_list[*]}; do 
            if [ ${autocast} = "amp" ]; then
                set_amp_config="Global.use_amp=True Global.scale_loss=1024.0 Global.use_dynamic_loss_scaling=True"
            else
                set_amp_config=" "
            fi          
            for trainer in ${trainer_list[*]}; do 
                flag_quant=False
                if [ ${trainer} = ${pact_key} ]; then
                    run_train=${pact_trainer}
                    run_export=${pact_export}
                    flag_quant=True
                elif [ ${trainer} = "${fpgm_key}" ]; then
                    run_train=${fpgm_trainer}
                    run_export=${fpgm_export}
                elif [ ${trainer} = "${distill_key}" ]; then
                    run_train=${distill_trainer}
                    run_export=${distill_export}
                elif [ ${trainer} = ${trainer_key1} ]; then
                    run_train=${trainer_value1}
                    run_export=${export_value1}
                elif [[ ${trainer} = ${trainer_key2} ]]; then
                    run_train=${trainer_value2}
                    run_export=${export_value2}
                else
                    run_train=${norm_trainer}
                    run_export=${norm_export}
                fi

                if [ ${run_train} = "null" ]; then
                    continue
                fi
                set_autocast=$(func_set_params "${autocast_key}" "${autocast}")
                set_epoch=$(func_set_params "${epoch_key}" "${epoch_num}")
                set_pretrain=$(func_set_params "${pretrain_model_key}" "${pretrain_model_value}")
                set_batchsize=$(func_set_params "${train_batch_key}" "${train_batch_value}")
                set_train_params1=$(func_set_params "${train_param_key1}" "${train_param_value1}")
                set_use_gpu=$(func_set_params "${train_use_gpu_key}" "${train_use_gpu}")
                if [ ${#ips} -le 26 ];then
                    nodes=1
                    save_log="${LOG_PATH}/${trainer}_gpus_${gpu}_autocast_${autocast}_nodes_${nodes}.log"
                else
                    IFS=","
                    ips_array=(${ips})
                    IFS="|"
                    nodes=${#ips_array[@]}
                    save_log="${LOG_PATH}/${trainer}_gpus_${gpu}_autocast_${autocast}_nodes_${nodes}.log"
                fi

                # load pretrain from norm training if current trainer is pact or fpgm trainer
                if ([ ${trainer} = ${pact_key} ] || [ ${trainer} = ${fpgm_key} ]) && [ ${nodes} -le 1 ]; then
                    set_pretrain="${load_norm_train_model}"
                fi

                set_save_model=$(func_set_params "${save_model_key}" "${save_log}")
                if [ ${#gpu} -le 2 ];then  # train with cpu or single gpu
                    cmd="${python} ${run_train} ${set_use_gpu}  ${set_save_model} ${set_epoch} ${set_pretrain} ${set_autocast} ${set_batchsize} ${set_train_params1} ${set_amp_config} "
                    eval "unset CUDA_VISIBLE_DEVICES"
                    train_log_path="${LOG_PATH}/${trainer}_gpus_${gpu}_autocast_${autocast}_nodes_${nodes}.log.log"
                    eval "${cmd} > ${train_log_path} 2>&1"
                    status_check $? "${cmd}" "${status_log}" "${model_name}" "${save_log}"

                elif [ ${#ips} -le 26 ];then  # train with multi-gpu
                    # run pserver
                    export TRAINING_ROLE=PSERVER
                    for((i=0;i<$PADDLE_PSERVER_NUMS;i++))
                    do
                        cur_port=${PADDLE_PSERVER_PORT_ARRAY[$i]}
                        echo "PADDLE WILL START PSERVER "$cur_port
                        export PADDLE_PORT=${cur_port}
                        cmd="${python} ${SC}"
                        eval "unset CUDA_VISIBLE_DEVICES"
                        eval $cmd
                        status_check $? "${cmd}" "${status_log}" "${model_name}" "${save_log}"
                    done

                    # run trainer
                    export TRAINING_ROLE=TRAINER
                    for((i=0;i<$PADDLE_TRAINERS;i++))
                    do
                        echo "PADDLE WILL START Trainer "$i
                        export PADDLE_TRAINER_ID=$i
                        cmd="${python} ${SC}"
                        eval "unset CUDA_VISIBLE_DEVICES"
                        eval $cmd
                        status_check $? "${cmd}" "${status_log}" "${model_name}" "${save_log}"
                    done
                else     # train with multi-machine
                    cmd="${python} -m paddle.distributed.launch --ips=${ips} --devices=${gpu} ${run_train} ${set_use_gpu} ${set_save_model} ${set_pretrain} ${set_epoch} ${set_autocast} ${set_batchsize} ${set_train_params1} ${set_amp_config}"
                    eval "unset CUDA_VISIBLE_DEVICES"
                    eval $cmd
                    status_check $? "${cmd}" "${status_log}" "${model_name}" "${save_log}"

                fi
                # run train

                set_eval_pretrain=$(func_set_params "${pretrain_model_key}" "${save_log}/${train_model_name}")
                # save norm trained models to set pretrain for pact training and fpgm training 
                if [ ${trainer} = ${trainer_norm} ] && [ ${nodes} -le 1 ]; then
                    load_norm_train_model=${set_eval_pretrain}
                fi
                # run eval 
                if [ ${eval_py} != "null" ]; then
                    set_eval_params1=$(func_set_params "${eval_key1}" "${eval_value1}")
                    eval_cmd="${python} ${eval_py} ${set_eval_pretrain} ${set_use_gpu} ${set_eval_params1}" 
                    eval $eval_cmd
                    status_check $? "${eval_cmd}" "${status_log}" "${model_name}" "${save_log}"
                fi
                # run export model
                if [ ${run_export} != "null" ]; then 
                    # run export model
                    save_infer_path="${save_log}"
                    set_export_weight=$(func_set_params "${export_weight}" "${save_log}/${train_model_name}")
                    set_save_infer_key=$(func_set_params "${save_infer_key}" "${save_infer_path}")
                    export_cmd="${python} ${run_export} ${set_export_weight} ${set_save_infer_key}"
                    eval $export_cmd
                    status_check $? "${export_cmd}" "${status_log}" "${model_name}" "${save_log}"

                    #run inference
                    eval $env
                    save_infer_path="${save_log}"
                    if [ "${inference_dir}" != "null" ] && [ "${inference_dir}" != '##' ]; then
                        infer_model_dir="${save_infer_path}/${inference_dir}"
                    else
                        infer_model_dir=${save_infer_path}
                    fi
                    func_inference "${python}" "${inference_py}" "${infer_model_dir}" "${LOG_PATH}" "${train_infer_img_dir}" "${flag_quant}"
                    
                    eval "unset CUDA_VISIBLE_DEVICES"
                fi
            done  # done with:    for trainer in ${trainer_list[*]}; do 
        done      # done with:    for autocast in ${autocast_list[*]}; do 
    done          # done with:    for gpu in ${gpu_list[*]}; do
fi  # end if [ ${MODE} = "infer" ]; then
