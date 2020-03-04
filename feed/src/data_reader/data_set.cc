#include "paddle/fluid/feed/src/data_reader/data_set.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/fluid/platform/timer.h"

namespace paddle {
namespace framework {

void FeedMultiSlotDataset::CreatePreLoadReaders() {
  VLOG(3) << "Begin CreatePreLoadReaders";
  if (preload_thread_num_ == 0) {
    preload_thread_num_ = thread_num_;
  }
  CHECK(preload_thread_num_ > 0) << "thread num should > 0";
  CHECK(input_channel_ != nullptr);
  preload_readers_.clear();
  for (int i = 0; i < preload_thread_num_; ++i) {
    preload_readers_.push_back(
        DataFeedFactory::CreateDataFeed(data_feed_desc_.name()));
    preload_readers_[i]->Init(data_feed_desc_);
    preload_readers_[i]->SetThreadId(i);
    preload_readers_[i]->SetThreadNum(preload_thread_num_);
    preload_readers_[i]->SetFileListMutex(&mutex_for_pick_file_);
    preload_readers_[i]->SetFileListIndex(&file_idx_);
    preload_readers_[i]->SetFileList(filelist_);
    preload_readers_[i]->SetParseInsId(parse_ins_id_);
    preload_readers_[i]->SetParseContent(parse_content_);
    preload_readers_[i]->SetInputChannel(input_channel_.get());
    preload_readers_[i]->SetOutputChannel(nullptr);
    preload_readers_[i]->SetConsumeChannel(nullptr);
  }
  VLOG(3) << "End CreatePreLoadReaders";
}

void FeedMultiSlotDataset::MergeByInsId() {
  VLOG(3) << "MultiSlotDataset::MergeByInsId begin";
  if (!merge_by_insid_) {
    VLOG(3) << "merge_by_insid=false, will not MergeByInsId";
    return;
  }
  auto multi_slot_desc = data_feed_desc_.multi_slot_desc();
  std::vector<std::string> use_slots;
  for (size_t i = 0; i < multi_slot_desc.slots_size(); ++i) {
    const auto& slot = multi_slot_desc.slots(i);
    if (slot.is_used()) {
      use_slots.push_back(slot.name());
    }
  }
  CHECK(multi_output_channel_.size() != 0);  // NOLINT
  auto channel_data = paddle::framework::MakeChannel<Record>();
  VLOG(3) << "multi_output_channel_.size() " << multi_output_channel_.size();
  for (size_t i = 0; i < multi_output_channel_.size(); ++i) {
    std::vector<Record> vec_data;
    multi_output_channel_[i]->Close();
    multi_output_channel_[i]->ReadAll(vec_data);
    channel_data->Write(std::move(vec_data));
    vec_data.clear();
    vec_data.shrink_to_fit();
    multi_output_channel_[i]->Clear();
  }
  channel_data->Close();
  std::vector<Record> recs;
  recs.reserve(channel_data->Size());
  channel_data->ReadAll(recs);
  channel_data->Clear();
  std::sort(recs.begin(), recs.end(), [](const Record& a, const Record& b) {
    return a.ins_id_ < b.ins_id_;
  });

  std::vector<Record> results;
  uint64_t drop_ins_num = 0;
  std::unordered_set<uint16_t> all_int64;
  std::unordered_set<uint16_t> all_float;
  std::unordered_set<uint16_t> local_uint64;
  std::unordered_set<uint16_t> local_float;

  VLOG(3) << "recs.size() " << recs.size();
  for (size_t i = 0; i < recs.size();) {
    size_t j = i + 1;
    while (j < recs.size() && recs[j].ins_id_ == recs[i].ins_id_) {
      j++;
    }
    if (min_merge_size_ > 0 && j - i != min_merge_size_) {
      drop_ins_num += j - i;
      LOG(WARNING) << "drop ins " << recs[i].ins_id_ << " size=" << j - i
                   << ", because merge_size=" << min_merge_size_;
      i = j;
      continue;
    }

    all_int64.clear();
    all_float.clear();
    bool has_conflict_slot = false;
    uint16_t conflict_slot = 0;

    Record rec;
    rec.ins_id_ = recs[i].ins_id_;
    rec.content_ = recs[i].content_;

    for (size_t k = i; k < j; k++) {
      local_uint64.clear();
      local_float.clear();
      for (auto& feature : recs[k].uint64_feasigns_) {
        uint16_t slot = feature.slot();
        if (all_int64.find(slot) != all_int64.end()) {
          has_conflict_slot = true;
          conflict_slot = slot;
          break;
        }
        local_uint64.insert(slot);
        rec.uint64_feasigns_.push_back(std::move(feature));
      }
      if (has_conflict_slot) {
        break;
      }
      all_int64.insert(local_uint64.begin(), local_uint64.end());

      for (auto& feature : recs[k].float_feasigns_) {
        uint16_t slot = feature.slot();
        if (all_float.find(slot) != all_float.end()) {
          has_conflict_slot = true;
          conflict_slot = slot;
          break;
        }
        local_float.insert(slot);
        rec.float_feasigns_.push_back(std::move(feature));
      }
      if (has_conflict_slot) {
        break;
      }
      all_float.insert(local_float.begin(), local_float.end());
    }

    if (has_conflict_slot) {
      LOG(WARNING) << "drop ins " << recs[i].ins_id_ << " size=" << j - i
                   << ", because conflict_slot=" << use_slots[conflict_slot];
      drop_ins_num += j - i;
    } else {
      results.push_back(std::move(rec));
    }
    i = j;
  }
  std::vector<Record>().swap(recs);
  VLOG(3) << "results size " << results.size();
  LOG(WARNING) << "total drop ins num: " << drop_ins_num;
  results.shrink_to_fit();

  auto fleet_ptr = FleetWrapper::GetInstance();
  std::shuffle(results.begin(), results.end(), fleet_ptr->LocalRandomEngine());
  channel_data->Open();
  channel_data->Write(std::move(results));
  channel_data->Close();
  results.clear();
  results.shrink_to_fit();
  VLOG(3) << "channel data size " << channel_data->Size();
  channel_data->SetBlockSize(channel_data->Size() / channel_num_ + 1);
  VLOG(3) << "channel data block size " << channel_data->BlockSize();
  for (size_t i = 0; i < multi_output_channel_.size(); ++i) {
    std::vector<Record> vec_data;
    channel_data->Read(vec_data);
    multi_output_channel_[i]->Open();
    multi_output_channel_[i]->Write(std::move(vec_data));
    vec_data.clear();
    vec_data.shrink_to_fit();
  }
  CHECK(channel_data->Size() == 0);  // NOLINT
  channel_data->Clear();
  VLOG(3) << "MultiSlotDataset::MergeByInsId end";
}

}  // end namespace framework
}  // end namespace paddle
