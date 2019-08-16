#pragma once

#include <map>
#include <string>
#include <iostream>
#include <vector>
#include <glog/logging.h>

namespace paddle {
namespace custom_trainer {
namespace feed {

class Any {
public:
    Any() : content_(NULL) {}

    template<typename ValueType>
        Any(const ValueType &value) : content_(new Holder<ValueType>(value)) {}

    Any(const Any &other) : content_(other.content_ ? other.content_->clone() : NULL) {}
    
    ~Any() {
        delete content_;
    }

    template<typename ValueType> ValueType *any_cast() {
        return content_ ? &static_cast<Holder<ValueType> *>(content_)->held_ : NULL;
    }

private:
    class PlaceHolder {
    public:
        virtual ~PlaceHolder() {}
        virtual PlaceHolder *clone() const = 0;
    };

    template<typename ValueType>
        class Holder : public PlaceHolder {
        public:
            explicit Holder(const ValueType &value) : held_(value) {}
            virtual PlaceHolder *clone() const {
                return new Holder(held_);
            }

            ValueType held_;
        };

    PlaceHolder *content_;
};

class ObjectFactory {
public:
    ObjectFactory() {}
    virtual ~ObjectFactory() {}
    virtual Any NewInstance() {
        return Any();
    }
private:
};

typedef std::map<std::string, ObjectFactory*> FactoryMap;
typedef std::map<std::string, FactoryMap> BaseClassMap;
#ifdef __cplusplus
extern "C" {
#endif
BaseClassMap& global_reg_factory_map();
#ifdef __cplusplus
}
#endif

BaseClassMap& global_reg_factory_map_cpp();

#define REGIST_REGISTERER(base_class) \
    class base_class ## Registerer { \
        public: \
            static base_class *CreateInstanceByName(const ::std::string &name) { \
                if (global_reg_factory_map_cpp().find(#base_class) \
                        == global_reg_factory_map_cpp().end()) { \
                    LOG(ERROR) << "Can't Find BaseClass For CreateClass with:" << #base_class; \
                    return NULL; \
                } \
                FactoryMap &map = global_reg_factory_map_cpp()[#base_class]; \
                FactoryMap::iterator iter = map.find(name); \
                if (iter == map.end()) { \
                    LOG(ERROR) << "Can't Find Class For Create with:" << name; \
                    return NULL; \
                } \
                Any object = iter->second->NewInstance(); \
                return *(object.any_cast<base_class*>()); \
            } \
    };

#define REGIST_CLASS(clazz, name) \
    class ObjectFactory##name : public ObjectFactory { \
        public: \
            Any NewInstance() { \
                return Any(new name()); \
            } \
    }; \
    void register_factory_##name() { \
        FactoryMap &map = global_reg_factory_map_cpp()[#clazz]; \
            if (map.find(#name) == map.end()) { \
                map[#name] = new ObjectFactory##name(); \
            } \
    } \
    void register_factory_##name() __attribute__((constructor)); 

#define CREATE_INSTANCE(base_class, name) \
    base_class##Registerer::CreateInstanceByName(name)
    
}//namespace feed
}//namespace custom_trainer
}//namespace paddle
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
