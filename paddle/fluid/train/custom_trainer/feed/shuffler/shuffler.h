#pragma once
#include "paddle/fluid/train/custom_trainer/feed/dataset/data_reader.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class TrainerContext;

inline double current_realtime() {
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    return tp.tv_sec + tp.tv_nsec * 1e-9;
}

inline std::default_random_engine& local_random_engine() {
    struct engine_wrapper_t {
        std::default_random_engine engine;
        engine_wrapper_t() {
            static std::atomic<unsigned long> x(0);
            std::seed_seq sseq = {x++, x++, x++, (unsigned long)(current_realtime() * 1000)};
            engine.seed(sseq);
        }
    };
    thread_local engine_wrapper_t r;
    return r.engine;
}

inline uint64_t shuffle_key_random(const std::string& /*key*/) {
    return local_random_engine()();
}   

inline uint64_t shuffle_key_hash(const std::string& key) {
    static std::hash<std::string> hasher;
    return hasher(key);
}   

inline uint64_t shuffle_key_numeric(const std::string& key) {
    return strtoull(key.c_str(), NULL, 10);
}

typedef uint64_t (*ShuffleKeyFunc)(const std::string& key);
inline ShuffleKeyFunc shuffle_key_factory(const std::string& name) {
    if (name == "NUMERIC") {
        return &shuffle_key_numeric;
    } else if (name == "HASH") {
        return &shuffle_key_hash;
    } 
    return &shuffle_key_random;
}


class Shuffler {
public:
    Shuffler() {}
    virtual ~Shuffler() {}
    virtual int initialize(YAML::Node config,
        std::shared_ptr<TrainerContext> context_ptr);
    virtual int shuffle(::paddle::framework::Channel<DataItem>& data_channel) = 0;
protected:
    ShuffleKeyFunc _shuffle_key_func;
    TrainerContext* _trainer_context;
};

REGIST_REGISTERER(Shuffler);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
