#pragma once
#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

namespace paddle {
namespace custom_trainer {
namespace feed {
    
class YamlHelper {
public:
    // 直接使用node["key"]判断，会导致node数据被加入key键
    static bool has_key(const YAML::Node& node, const std::string& key) {
        CHECK(node.Type() == YAML::NodeType::Map);
        for (const auto& itr : node) {
            if (key == itr.first.as<std::string>()) {
                return true;
            }
        } 
        return false;
    }
    template <class T>
    static T get_with_default(YAML::Node node, const std::string& key, const T& default_v) {
        if (has_key(node, key)) {
            return node[key].as<T>();
        }
        return default_v;
    }
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
