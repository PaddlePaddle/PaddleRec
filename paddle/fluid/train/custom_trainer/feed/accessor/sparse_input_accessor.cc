#include <vector>
#include <utility>
#include "paddle/fluid/string/string_helper.h"
#include "paddle/fluid/train/custom_trainer/feed/accessor/input_data_accessor.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class CommonSparseInputAccessor : public DataInputAccessor {
public:
    CommonSparseInputAccessor() {}
    virtual ~CommonSparseInputAccessor() {}
    virtual int initialize(const YAML::Node& config,
        std::shared_ptr<TrainerContext> context_ptr) {
        CHECK(config["sparse_input"] && config["sparse_input"].Type() == YAML::NodeType::Map);
        for (auto& input : config["sparse_input"]) {
            std::pair<std::string, std::vector<uint16_t>> sparse_slots;
            sparse_slots.first = input.first.as<std::string>();
            std::string slots_str = input.second["slots"].as<std::string>();
            std::vector<std::string> slots = paddle::string::split_string(slots_str, ","); 
            for (int i = 0; i < slots.size(); ++i) {
                sparse_slots.second.push_back((uint16_t)atoi(slots[i].c_str()));
            }
        }
        return 0;
    }

    // 取sparse数据
    virtual int32_t forward(const SampleInstance* samples,
        ::paddle::framework::Scope* scope, size_t table_id, size_t num) {
        // pull
        return 0;
    }

    // 更新spare数据
    virtual int32_t backward(const SampleInstance* samples,
        ::paddle::framework::Scope* scope, size_t table_id, size_t num) {
        return 0;
    } 

protected:

    // 输入层列表
    // <data_name, slot_id_list>
    std::vector<std::pair<std::string, std::vector<uint16_t> > > _x_variables;       
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
