#include "paddle/fluid/train/custom_trainer/feed/common/registerer.h"
namespace paddle {
namespace custom_trainer {
namespace feed {

BaseClassMap& global_reg_factory_map() {
    static BaseClassMap *base_class = new BaseClassMap();
    return *base_class;
}
BaseClassMap& global_reg_factory_map_cpp() {
    return global_reg_factory_map();
}

}// feed
}// namespace custom_trainer
}// namespace paddle
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

