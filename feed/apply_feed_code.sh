#!bash

#将FEED定制化代码生效到Paddle代码库（如FEED插件注册） 编译前执行


function fatal_log() {
    echo "$1"
    exit -1
}

#处理pybind 拓展
function apply_pybind() {
    pybind_file='paddle/fluid/pybind/pybind.cc'
    if [ ! -f ${pybind_file} ];then
        fatal_log "Missing Requied File:${pybind_file}"
    fi 

    find_inferece_api=`grep 'inference_api.h' ${pybind_file} |wc -l`
    if [ ${find_inferece_api} -ne 1 ];then
        fatal_log "Missing inference_api.h, Need Code Adjust"
    fi
    find_inferece_api=`grep 'BindInferenceApi' ${pybind_file} |wc -l`
    if [ ${find_inferece_api} -ne 1 ];then
        fatal_log "Missing BindInferenceApi, Need Code Adjust"
    fi

    makefile='paddle/fluid/pybind/CMakeLists.txt'
    if [ ! -f ${makefile} ];then
        fatal_log "Missing Requied File:${makefile}"
    fi

    sed -i '/expand_api/d' ${pybind_file}
    sed -i '/BindExpandApi/d' ${pybind_file}
    sed -i '/feed_data_set/d' ${makefile}
    sed -i '/feed_paddle_pybind/d' ${makefile}
    sed -i '/APPEND PYBIND_DEPS fs/d' ${makefile}

    sed -i '/inference_api.h/a\#include "paddle/fluid/feed/pybind/expand_api.h"' ${pybind_file}
    sed -i '/BindInferenceApi/a\  BindExpandApi(&m);' ${pybind_file}
    sed -i '/set(PYBIND_SRCS/i\list(APPEND PYBIND_DEPS feed_data_set)' ${makefile}
    sed -i '/set(PYBIND_SRCS/i\list(APPEND PYBIND_DEPS feed_paddle_pybind)' ${makefile}
    sed -i '/set(PYBIND_SRCS/i\list(APPEND PYBIND_DEPS fs)' ${makefile}
}

function apply_feed_src() {
    makefile='paddle/fluid/CMakeLists.txt'
    if [ ! -f ${makefile} ];then
        fatal_log "Missing Requied File:${makefile}"
    fi
    find_py=`grep 'pybind' ${makefile} |wc -l`
    if [ ${find_py} -ne 1 ];then
        fatal_log "Missing pybind, Need Code Adjust"
    fi
    sed -i '/feed/d' ${makefile}
    sed -i '/pybind/i\add_subdirectory(feed)' ${makefile}

    dataset_file='paddle/fluid/framework/dataset_factory.cc'
    if [ ! -f ${dataset_file} ];then
        fatal_log "Missing Requied File:${dataset_file}"
    fi
    sed -i '/FeedMultiSlotDataset/d' ${dataset_file}
    sed -i '/data_reader/d' ${dataset_file}
    sed -i '/REGISTER_DATASET_CLASS(MultiSlotDataset)/a\REGISTER_DATASET_CLASS(FeedMultiSlotDataset);' ${dataset_file}
    sed -i '/data_set.h/a\#include "paddle/fluid/feed/src/data_reader/data_set.h"' ${dataset_file}
    sed -i '/feed_data_set/d' paddle/fluid/framework/CMakeLists.txt
    #sed -i '/target_link_libraries(executor/a\target_link_libraries(feed_data_set)' paddle/fluid/framework/CMakeLists.txt
    #sed -i '/target_link_libraries(executor/a\add_dependencies(feed_data_set)' paddle/fluid/framework/CMakeLists.txt
}

apply_pybind
apply_feed_src

