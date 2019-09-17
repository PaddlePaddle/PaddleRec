#include <gtest/gtest.h>
#include "paddle/fluid/train/custom_trainer/feed/dataset/data_reader.h"

TEST(Archive, DataItem) {
    paddle::custom_trainer::feed::DataItem item;
    paddle::custom_trainer::feed::DataItem item2;
    item.id = "123";
    item.data = "name";

    paddle::framework::BinaryArchive ar;
    ar << item;
    ar >> item2;

    ASSERT_EQ(item.id, item2.id);
    ASSERT_EQ(item.data, item2.data);
    item.id += "~";
    item.data += "~";
    ASSERT_NE(item.id, item2.id);
    ASSERT_NE(item.data, item2.data);
}