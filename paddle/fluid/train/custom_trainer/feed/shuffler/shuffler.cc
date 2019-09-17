#include "paddle/fluid/framework/archive.h"
#include "paddle/fluid/train/custom_trainer/feed/trainer_context.h"
#include "paddle/fluid/train/custom_trainer/feed/shuffler/shuffler.h"
#include <bthread/butex.h>

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
REGIST_CLASS(Shuffler, LocalShuffler);

class GlobalShuffler : public Shuffler {
public:
    GlobalShuffler() {}
    virtual ~GlobalShuffler() {}
    virtual int initialize(YAML::Node config,
        std::shared_ptr<TrainerContext> context_ptr) {
        Shuffler::initialize(config, context_ptr);
        _max_concurrent_num = config["max_concurrent_num"].as<int>(4); // 最大并发发送数
        _max_package_size = config["max_package_size"].as<int>(1024);  // 最大包个数，一次发送package个数据
        _shuffle_data_msg_type = config["shuffle_data_msg_type"].as<int>(3);  // c2c msg type
        _finish_msg_type = config["finish_msg_type"].as<int>(4);  // c2c msg type

        reset_channel();
        auto binded = std::bind(&GlobalShuffler::get_client2client_msg, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
        _trainer_context->pslib->ps_client()->registe_client2client_msg_handler(_shuffle_data_msg_type,
            binded);
        _trainer_context->pslib->ps_client()->registe_client2client_msg_handler(_finish_msg_type,
            binded);
        return 0;
    }

    // 所有worker必须都调用shuffle，并且shuffler同时只能有一个shuffle任务
    virtual int shuffle(::paddle::framework::Channel<DataItem>& data_channel) {
        uint32_t send_count = 0;
        uint32_t package_size = _max_package_size;
        uint32_t concurrent_num = _max_concurrent_num;
        ::paddle::framework::Channel<DataItem> input_channel = ::paddle::framework::MakeChannel<DataItem>(data_channel);
        data_channel.swap(input_channel);
        set_channel(data_channel);

        auto* environment = _trainer_context->environment.get();
        auto worker_num = environment->node_num(EnvironmentRole::WORKER);
        std::vector<std::vector<std::future<int>>> waits(concurrent_num);
        std::vector<DataItem> send_buffer(package_size);
        std::vector<std::vector<DataItem>> send_buffer_worker(worker_num);

        int status = 0;// >0: finish; =0: running; <0: fail
        while (status == 0) {
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
            for (uint32_t current_wait_idx = 0; status == 0 && current_wait_idx < concurrent_num; ++current_wait_idx) {
                auto read_size = input_channel->Read(package_size, send_buffer.data());
                if (read_size == 0) {
                    status = 1;
                    break;
                }
                for (int i = 0; i < worker_num; ++i) {
                    send_buffer_worker.clear();
                }
                for (int i = 0; i < read_size; ++i) {
                    auto worker_idx = _shuffle_key_func(send_buffer[i].id) % worker_num;
                    send_buffer_worker[worker_idx].push_back(std::move(send_buffer[i]));
                }
                for (auto& wait_s : waits[current_wait_idx]) {
                    if (wait_s.get() != 0) {
                        LOG(WARNING) << "fail to send shuffle data";
                        status = -1;
                        break;
                    }
                }
                if (status != 0) {
                    break;
                }
                waits[current_wait_idx].clear();
                for (int i = 0; i < worker_num; ++i) {
                    if (!send_buffer_worker[i].empty()) {
                        waits[current_wait_idx].push_back(send_shuffle_data(i, send_buffer_worker[i]));
                    }
                }
            }
        }
        for (auto& waits_s : waits) {
            for (auto& wait_s : waits_s) {
                if (wait_s.get() != 0) {
                    LOG(WARNING) << "fail to send shuffle data";
                    status = -1;
                }
            }
        }
        VLOG(5) << "start send finish, worker_num: " << worker_num;
        waits[0].clear();
        for (int i = 0; i < worker_num; ++i) {
             waits[0].push_back(send_finish(i));
        }
        VLOG(5) << "wait all finish";
        for (int i = 0; i < worker_num; ++i) {
            if (waits[0][i].get() != 0) {
                LOG(WARNING) << "fail to send finish " << i;
                status = -1;
            }
        }
        VLOG(5) << "finish shuffler, status: " << status;
        return status < 0 ? status : 0;
    }

private:
    /*
    1. 部分c2c send_shuffle_data先到, 此时channel未设置, 等待wait_channel
    2. shuffle中调用set_channel, 先reset_wait_num, 再解锁channel
    3. 当接收到所有worker的finish请求后，先reset_channel, 再同时返回
    */
    bool wait_channel() {
        VLOG(5) << "wait_channel";
        std::lock_guard<bthread::Mutex> lock(_channel_mutex);
        return _out_channel != nullptr;
    }
    void reset_channel() {
        VLOG(5) << "reset_channel";
        _channel_mutex.lock();
        if (_out_channel != nullptr) {
            _out_channel->Close();
        }
        _out_channel = nullptr;
    }
    void reset_wait_num() {
        _wait_num_mutex.lock();
        _wait_num = _trainer_context->environment->node_num(EnvironmentRole::WORKER);
        VLOG(5) << "reset_wait_num: " << _wait_num;
    }
    void set_channel(paddle::framework::Channel<DataItem>& channel) {
        VLOG(5) << "set_channel";
        // 在节点开始写入channel之前，重置wait_num
        CHECK(_out_channel == nullptr);
        _out_channel = channel;
        reset_wait_num();
        _channel_mutex.unlock();
    }

    int32_t finish_write_channel() {
        int wait_num = --_wait_num;
        VLOG(5) << "finish_write_channel, wait_num: " << wait_num;
        // 同步所有worker，在所有写入完成后，c2c_msg返回前，重置channel
        if (wait_num == 0) {
            reset_channel();
            _wait_num_mutex.unlock();
        } else {
            std::lock_guard<bthread::Mutex> lock(_wait_num_mutex);
        }
        return 0;
    }
    int32_t write_to_channel(std::vector<DataItem>&& items) {
        size_t items_size = items.size();
        VLOG(5) << "write_to_channel, items_size: " << items_size;
        return _out_channel->Write(std::move(items)) == items_size ? 0 : -1;
    }

    int32_t get_client2client_msg(int msg_type, int from_client, const std::string& msg) {
        // wait channel
        if (!wait_channel()) {
            LOG(FATAL) << "out_channel is null";
            return -1;
        }
        VLOG(5) << "get c2c msg, type: " << msg_type << ", from_client: " << from_client << ", msg_size: " << msg.size();
        if (msg_type == _shuffle_data_msg_type) {
            paddle::framework::BinaryArchive ar;
            ar.SetReadBuffer(const_cast<char*>(msg.data()), msg.size(), [](char*){});
            std::vector<DataItem> items;
            ar >> items;
            return write_to_channel(std::move(items));
        } else if (msg_type == _finish_msg_type) {
            return finish_write_channel();
        }
        LOG(FATAL) << "no such msg type: " << msg_type;
        return -1;
    }

    std::future<int32_t> send_shuffle_data(int to_client_id, std::vector<DataItem>& items) {
        VLOG(5) << "send_shuffle_data, to_client_id: " << to_client_id << ", items_size: " << items.size();
        paddle::framework::BinaryArchive ar;
        ar << items;
        return _trainer_context->pslib->ps_client()->send_client2client_msg(_shuffle_data_msg_type, to_client_id,
            std::string(ar.Buffer(), ar.Length()));
    }

    std::future<int32_t> send_finish(int to_client_id) {
        VLOG(5) << "send_finish, to_client_id: " << to_client_id;
        static const std::string empty_str;
        return _trainer_context->pslib->ps_client()->send_client2client_msg(_finish_msg_type, to_client_id, empty_str);
    }

    uint32_t _max_package_size = 0;
    uint32_t _max_concurrent_num  = 0;
    int _shuffle_data_msg_type = 3;
    int _finish_msg_type = 4;
    
    bthread::Mutex _channel_mutex;
    paddle::framework::Channel<DataItem> _out_channel = nullptr;

    bthread::Mutex _wait_num_mutex;
    std::atomic<int> _wait_num;
};
REGIST_CLASS(Shuffler, GlobalShuffler);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
