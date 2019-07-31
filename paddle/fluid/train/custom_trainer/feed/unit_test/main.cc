#include <gtest/gtest.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/pybind/pybind.h"

int32_t main(int32_t argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc, argv);
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    return RUN_ALL_TESTS();
}
