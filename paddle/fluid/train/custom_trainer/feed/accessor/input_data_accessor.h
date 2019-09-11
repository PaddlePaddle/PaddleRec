#pragma once
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/train/custom_trainer/feed/accessor/accessor.h"
#include "paddle/fluid/train/custom_trainer/feed/dataset/data_reader.h"
#include "paddle/fluid/train/custom_trainer/feed/common/scope_helper.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class DataInputAccessor : public Accessor {
public:
    DataInputAccessor() {}
    virtual ~DataInputAccessor() {}

    virtual int initialize(YAML::Node config,
        std::shared_ptr<TrainerContext> context_ptr) {
        _trainer_context = context_ptr.get();
        _table_id = config["table_id"].as<int>();
        _need_gradient = config["need_gradient"].as<bool>();
        return 0;
    }

    // 创建，一般用于模型冷启的随机初始化
    virtual int32_t create(::paddle::framework::Scope* scope) {
        return 0;
    }
    // 裁剪，用于模型裁剪，base级调用
    virtual int32_t shrink() {
        return 0;
    }

    // 前向， 一般用于填充输入，在训练网络执行前调用
    virtual int32_t forward(SampleInstance* samples, size_t num,
        ::paddle::framework::Scope* scope) = 0;

    // 后向，一般用于更新梯度，在训练网络执行后调用, 由于backward一般是异步，这里返回future， 
    // TODO 前向接口也改为future返回形式，接口一致性好些
    virtual std::future<int32_t> backward(SampleInstance* samples, size_t num,
        ::paddle::framework::Scope* scope) = 0;

    // 收集持久化变量的名称, 并将值拷贝到Scope
    virtual int32_t collect_persistables_name(std::vector<std::string>& persistables) {return 0;}

    // 填充持久化变量的值，用于保存
    virtual int32_t collect_persistables(paddle::framework::Scope* scope) {return 0;}
protected:
    size_t _table_id = 0;
    bool _need_gradient = false;
    TrainerContext* _trainer_context = nullptr;
};
REGIST_REGISTERER(DataInputAccessor);

struct LabelInputVariable {
    std::string label_name;
    std::string output_name;
    size_t label_dim = 0;
};
class LabelInputAccessor : public DataInputAccessor {
public:
    LabelInputAccessor() {}
    virtual ~LabelInputAccessor() {}
    
    virtual int initialize(YAML::Node config,
         std::shared_ptr<TrainerContext> context_ptr);

    virtual int32_t forward(SampleInstance* samples, size_t num,
        ::paddle::framework::Scope* scope);

    virtual std::future<int32_t> backward(SampleInstance* samples, size_t num,
        ::paddle::framework::Scope* scope);
protected:
    size_t _label_total_dim = 0; 
    std::vector<LabelInputVariable> _labels;
};

struct SparseInputVariable {
    size_t slot_dim;
    size_t total_dim;
    std::string name;
    std::string gradient_name;
    std::vector<int32_t> slot_idx;
    std::vector<uint16_t> slot_list;
};

struct SparseVarRuntimeData {
    uint32_t row_size;
    uint32_t total_size;
    float* variable_data;
    float* gradient_data;
};

class BaseSparseInputAccessor : public DataInputAccessor {
public:
    BaseSparseInputAccessor() {}
    virtual ~BaseSparseInputAccessor() {}

    virtual int initialize(YAML::Node config,
        std::shared_ptr<TrainerContext> context_ptr);

    // forword过程的input填充
    virtual int32_t forward(SampleInstance* samples, size_t num,
        paddle::framework::Scope* scope);
    // 取得单个SparseKey的PullValue, 实现单个SparseValue的填充
    virtual void fill_input(float* var_data, const float* pull_raw,
        paddle::ps::ValueAccessor&, SparseInputVariable&, SampleInstance&) = 0;
    // 所有SparseValue填充完成后，调用，可进一步全局处理
    virtual void post_process_input(float* var_data, SparseInputVariable&, SampleInstance*, size_t num) = 0;

    // backward过程的梯度push
    virtual std::future<int32_t> backward(SampleInstance* samples, size_t num,
        paddle::framework::Scope* scope);    
    // SparseGradValue会被依次调用，用于整理push的梯度
    virtual void fill_gradient(float* push_value, const float* gradient_raw, 
        paddle::ps::ValueAccessor&, SparseInputVariable&, 
        SampleInstance&, FeatureItem&) = 0;

protected:
    // 输入层列表
    std::vector<SparseInputVariable> _x_variables;       
};

struct DenseInputVariable {
    size_t dim;
    std::string name;
    std::vector<int> shape;
    std::string gradient_name;
};

class DenseInputAccessor : public DataInputAccessor {
public:
    DenseInputAccessor() {}
    virtual ~DenseInputAccessor() {
        for (float* buffer : _data_buffer_list) {
            delete[] buffer;
        }
        _need_async_pull = false;
        if (_async_pull_thread) {
            _async_pull_thread->join();
        }
    }

    // 返回当前可用的Dense buffer
    inline float* data_buffer() {
        return _data_buffer_list[_current_buffer_idx];
    }
    inline float* backend_data_buffer() {
        return _data_buffer_list[next_buffer_idx()];
    }
    inline void switch_data_buffer() {
        _current_buffer_idx = next_buffer_idx(); 
    }
    inline size_t next_buffer_idx() {
        auto buffer_idx = _current_buffer_idx + 1;
        if (buffer_idx >= _data_buffer_list.size()) {
            return 0;
        }
        return buffer_idx;
    }
    
    virtual int initialize(YAML::Node config,
        std::shared_ptr<TrainerContext> context_ptr);
    
    virtual int32_t create(::paddle::framework::Scope* scope);

    virtual int32_t forward(SampleInstance* samples, size_t num,
        paddle::framework::Scope* scope);

    virtual std::future<int32_t> backward(SampleInstance* samples, size_t num,
        paddle::framework::Scope* scope);


    virtual int32_t collect_persistables_name(std::vector<std::string>& persistables);

    virtual int32_t collect_persistables(paddle::framework::Scope* scope);
protected:
    virtual int32_t pull_dense(size_t table_id);
    size_t _total_dim = 0;
    std::mutex _pull_mutex;
    bool _need_async_pull = false;
    bool _is_data_buffer_init = false;
    std::vector<float*> _data_buffer_list;
    size_t _current_buffer_idx = 0;
    std::atomic<int> _pull_request_num;
    std::vector<DenseInputVariable> _x_variables; 
    std::shared_ptr<std::thread> _async_pull_thread;
};

class EbdVariableInputAccessor : public DenseInputAccessor {
public:
    EbdVariableInputAccessor() {}
    virtual ~EbdVariableInputAccessor() {}

    virtual int32_t forward(SampleInstance* samples, size_t num,
        paddle::framework::Scope* scope);

    virtual std::future<int32_t> backward(SampleInstance* samples, size_t num,
        paddle::framework::Scope* scope);
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
