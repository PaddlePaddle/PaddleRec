day=20190720
pass_id=6
ROOT_PATH=/xxx/PaddleRec
TOOL_PATH=$ROOT_PATH/tools/onoff_diff
MODEL_PATH=$ROOT_PATH/models/rank/slot_dnn
# INFERENCE_PATH=$ROOT_PATH/tools/inference_c++2.0
# 在线预测输入数据
ONLINE_FILE=$MODEL_PATH/infer_data/online/demo_10
# 离线预测输入数据
OFFLINE_FILE=$MODEL_PATH/infer_data/offline/demo_10
# 在线预测输入数据添加insid作为离线预测输入数据
echo "get offline data..."
cat $ONLINE_FILE | awk -F'\t' 'BEGIN{OFS="\t"}{print NR,$0}' > $OFFLINE_FILE

# 在线预测
# 在线预测数据放在$MODEL_PATH/infer_data/online中，无insid
# cd $INFERENCE_PATH
# ./bin/main --flagfile ./user.flags
# cp std.log $TOOL_PATH/data/log.online

# 离线预测
CUBE_FILE=./data/cube.result
XBOX_FILE=./data/xbox_model_result
SAVE_MODEL_PATH=$MODEL_PATH/output_model/$day/inference_model_$pass_id/
# cp online.cube.result to data/cube.result
# 将cube输出文件修改成大模型格式，并放入离线预测加载模型路径
echo "get xbox model..."
cd $TOOL_PATH
python3 get_xbox_model.py -i $CUBE_FILE \
                          -o $XBOX_FILE
sort -u  $XBOX_FILE > $SAVE_MODEL_PATH/embedding.shard/embedding.block0.txt

# copy小模型

# 离线预测数据放在$MODEL_PATH/infer_data/offline中，添加insid
echo "offline predict..."
cd $MODEL_PATH
fleetrun --server_num=1 --worker_num=1 ../../../tools/static_ps_offline_infer.py -m config_offline_infer.yaml 
OFFLINE_DUMP_PATH=$MODEL_PATH/dump_offline_infer
cat $OFFLINE_DUMP_PATH/part* > $TOOL_PATH/data/log.offline

echo "online offline diff..." 
cd $TOOL_PATH
python3 onoff_diff.py -l ./data/log.online \
                      -m ./data/log.offline \
                      -v $MODEL_PATH/all_vars.txt \
                      -o $OFFLINE_FILE
