#!bash
OUTPUT_PATH=../../../bc_out/baidu/feed-mlarch/paddle-trainer/output/include/
INCLUDE_DIR=paddle/fluid/train/custom_trainer/feed/
SUB_DIR_LIST=(common dataset accessor executor monitor process shuffler)
rm -rf ${OUTPUT_PATH}/${INCLUDE_DIR}/*

cp ${INCLUDE_DIR}/*.h ${OUTPUT_PATH}/${INCLUDE_DIR}/
for sub_name in "${SUB_DIR_LIST[@]}"
do
    mkdir ${OUTPUT_PATH}/${INCLUDE_DIR}/${sub_name}
    cp ${INCLUDE_DIR}/${sub_name}/*.h ${OUTPUT_PATH}/${INCLUDE_DIR}/${sub_name}/
done
