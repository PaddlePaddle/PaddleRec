add_subdirectory(common)
add_subdirectory(process)
cc_library(custom_trainer_main SRCS main.cc DEPS custom_trainer_process custom_trainer_common)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(ARCHIVE_START "-Wl,--whole-archive")
set(ARCHIVE_END "-Wl,--no-whole-archive")
set(EXTERNAL_LIB "-lrt -ldl -lpthread")

add_executable(feed_trainer main.cc)
target_link_libraries(feed_trainer
        ${MACOS_LD_FLAGS}
        ${ARCHIVE_START}
        ${ARCHIVE_END}
        glog gflags protobuf snappystream snappy z xxhash yaml-cpp 
        paddle_fluid custom_trainer_common custom_trainer_process
        ${EXTERNAL_LIB})
