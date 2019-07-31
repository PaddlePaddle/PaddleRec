#include <gtest/gtest.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

int32_t main(int32_t argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging("paddle_trainer");
    return RUN_ALL_TESTS();
}
