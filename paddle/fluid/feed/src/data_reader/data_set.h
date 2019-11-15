#pragma once
#include "paddle/fluid/framework/data_set.h"

namespace paddle {
namespace framework {

class FeedMultiSlotDataset : public MultiSlotDataset {
 public:
  FeedMultiSlotDataset() {}
  virtual void MergeByInsId();
  virtual void CreatePreLoadReaders();
  virtual ~FeedMultiSlotDataset() {}
};

}  // end namespace framework
}  // end namespace paddle
