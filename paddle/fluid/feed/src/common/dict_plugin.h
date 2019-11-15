#pragma once

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <glog/logging.h>
#include "paddle/fluid/feed/src/common/bhopscotch_map.h"

namespace paddle {
namespace framework {
class DictPlugin {
public:
    DictPlugin() {}
    virtual ~DictPlugin() {}
    virtual int Load(const std::string& path, const std::string& converter) = 0;
};

template <class K, class V>
class KvEntity {
public:
    KvEntity() {}
    ~KvEntity() {}
    uint32_t Size() {
        return _key_list.size();
    }
    void Append(const K& k, const V& v) {
        if (_dict_data.find(k) != _dict_data.end()) {
            return;
        }
        _key_list.push_back(k);
        _dict_data.emplace(k, v);
    }
    std::vector<K> _key_list;
    tsl::bhopscotch_pg_map<K, V> _dict_data; 
};

template <class K, class V>
class KvDictPlugin : public DictPlugin {
public:
    KvDictPlugin() {
        versioned_entity_.resize(2);
    }
    virtual ~KvDictPlugin() {} 
    
    // GetValue with version, Return: value
    virtual int GetValueWithVersion(uint32_t version, const K& key, V& v) {
        CHECK(version < versioned_entity_.size());
        auto& entity = versioned_entity_[version];
        auto itr = entity._dict_data.find(key);
        if (itr == entity._dict_data.end()) {
            return -1; // miss
        }
        v = itr->second;
        return 0;
    }

    // GetValue without version, Return: value version
    virtual int GetValue(const K& key, V& v, uint32_t& version) {
        version = version_;
        auto& entity = versioned_entity_[version];
        auto itr = entity._dict_data.find(key);
        if (itr == entity._dict_data.end()) {
            return -1; // miss
        }
        v = itr->second;
        return 0;
    }
    
    virtual int GetVersion() {
        return version_;
    }
protected:
    uint32_t version_ = 0;
    // double-buffer support version:0 1
    std::vector<KvEntity<K, V>> versioned_entity_;
};

class FeasignCacheDict : public KvDictPlugin<uint64_t, uint32_t> {
public:
    FeasignCacheDict(){}
    virtual ~FeasignCacheDict(){}
    virtual int Load(const std::string& path, const std::string& converter);
};

class DictPluginManager {
public:
    DictPluginManager() {}
    virtual ~DictPluginManager(){}

    static DictPluginManager& Instance() {
        static DictPluginManager manager;
        return manager;
    }
    inline int CreateDict(const std::string& dict_name) {
        #define PADDLE_DICT_PLUGIN_REGIST(dict)                  \
        if (dict_name == #dict) {                                \
            dicts_map_[dict_name].reset(new dict());             \
            return 0;                                            \
        }
        
        PADDLE_DICT_PLUGIN_REGIST(FeasignCacheDict)
        #undef PADDLE_DICT_PLUGIN_REGIST
        return -1;
    }
    inline DictPlugin* GetDict(const std::string& dict_name) {
        if (dicts_map_.count(dict_name)) {
            return dicts_map_[dict_name].get();
        }
        return nullptr;
    }
    inline int LoadDict(const std::string& dict_name,
        const std::string& path, const std::string converter) {
        auto dict = GetDict(dict_name);
        if (!dict) {
            return -1;
        }
        return dict->Load(path, converter);
    }
private:
    std::unordered_map<std::string, std::shared_ptr<DictPlugin>> dicts_map_;
};

}  // namespace framework
}  // namespace paddle
