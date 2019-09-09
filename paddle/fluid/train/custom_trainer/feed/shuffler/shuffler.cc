#pragma once
#include "paddle/fluid/framework/archive.h"
#include "paddle/fluid/train/custom_trainer/feed/trainer_context.h"
#include "paddle/fluid/train/custom_trainer/feed/shuffler/shuffler.h"

namespace paddle {
namespace custom_trainer {
namespace feed {


int Shuffler::initialize(YAML::Node config,
    std::shared_ptr<TrainerContext> context_ptr) {
    _trainer_context = context_ptr.get();
    _shuffle_key_func = shuffle_key_factory(config["shuffle_key_func"].as<std::string>("RANDOM"));
    return 0;
}

class LocalShuffler : public Shuffler {
public:
    LocalShuffler() {}
    virtual ~LocalShuffler() {}
    virtual int shuffle(::paddle::framework::Channel<DataItem>& data_channel) {
        std::vector<DataItem> data_items(data_channel->Size());
        data_channel->ReadAll(data_items);
        std::shuffle(data_items.begin(), data_items.end(), local_random_engine());
        data_channel->Open();
        data_channel->Clear();
        data_channel->WriteMove(data_items.size(), &data_items[0]);
        data_channel->Close();
        return 0;
    }
};
REGIST_CLASS(DataParser, LocalShuffler);

class GlobalShuffler : public Shuffler {
public:
    GlobalShuffler() {}
    virtual ~GlobalShuffler() {}
    virtual int initialize(YAML::Node config,
        std::shared_ptr<TrainerContext> context_ptr) {
        Shuffler::initialize(config, context_ptr);
        _max_concurrent_num = config["max_concurrent_num"].as<int>(4); // 最大并发发送数
        _max_package_size = config["max_package_size"].as<int>(1024);  // 最大包个数，一次发送package个数据
        return 0;
    }
    virtual int shuffle(::paddle::framework::Channel<DataItem>& data_channel) {
        uint32_t send_count = 0;
        uint32_t package_size = _max_package_size;
        uint32_t concurrent_num = _max_concurrent_num;
        uint32_t current_wait_idx = 0;
        auto* environment = _trainer_context->environment.get();
        auto worker_num = environment->node_num(EnvironmentRole::WORKER);
        std::vector<std::vector<std::future<int>>> waits(concurrent_num);
        std::vector<DataItem> send_buffer(concurrent_num * package_size);
        std::vector<paddle::framework::BinaryArchive> request_data_buffer(worker_num);
        while (true) {
            auto read_size = data_channel->Read(concurrent_num * package_size, &send_buffer[0]);
            if (read_size == 0) {
                break;
            }
            for (size_t idx = 0; idx < read_size; idx += package_size) {
                // data shard && seriliaze
                for (size_t i = 0; i < worker_num; ++i) {
                    request_data_buffer[i].Clear();
                }
                for (size_t i = idx; i < package_size && i < read_size; ++i) {
                    auto worker_idx = _shuffle_key_func(send_buffer[i].id) % worker_num;
                    // TODO Serialize To Arcive
                    //request_data_buffer[worker_idx] << send_buffer[i];
                }
                std::string data_vec[worker_num];
                for (size_t i = 0; i < worker_num; ++i) {
                    auto& buffer = request_data_buffer[i];
                    data_vec[i].assign(buffer.Buffer(), buffer.Length()); 
                }

                // wait async done
                for (auto& wait_s : waits[current_wait_idx]) {
                    if (!wait_s.valid()) {
                        break;
                    }
                    CHECK(wait_s.get() == 0);
                }

                // send shuffle data
                for (size_t i = 0; i < worker_num; ++i) {
                    waits[current_wait_idx][i] = _trainer_context->pslib->ps_client()->send_client2client_msg(3, i * 2, data_vec[i]);
                }
                
                // update status
                // 如果在训练期，则限速shuffle     
                // 如果在wait状态，全速shuffle
                if (_trainer_context->is_status(TrainerStatus::Training)) {
                    concurrent_num = 1;
                    package_size = _max_concurrent_num / 2;
                } else {
                    package_size = _max_package_size;
                    concurrent_num = _max_concurrent_num;
                }
                ++current_wait_idx;
                current_wait_idx = current_wait_idx >= concurrent_num ? 0 : current_wait_idx;
            }
        }
        return 0;
    }

private:
    uint32_t _max_package_size = 0;
    uint32_t _max_concurrent_num  = 0;
    
};
REGIST_CLASS(DataParser, GlobalShuffler);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
