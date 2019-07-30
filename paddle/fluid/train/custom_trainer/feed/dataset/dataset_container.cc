/* DatasetContainer
 * 保存一个数据源的样本，并驱动样本的异步加载
 */
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <yaml-cpp/yaml.h>
#include "paddle/fluid/framework/io/shell.h"
#include "paddle/fluid/string/string_helper.h"
#include "paddle/fluid/train/custom_trainer/feed/dataset/dataset_container.h"

namespace paddle {
namespace custom_trainer {
namespace feed {
     
paddle::framework::Channel<DataItem> DatasetContainer::fetch(int epoch_id) {
    paddle::framework::Channel<DataItem> result;
    if (_ready_epoch_id < epoch_id) {
        return result;
    }
    _current_epoch_id = epoch_id;
    _current_dataset_idx = epoch_id % _prefetch_num;
    //result = _dataset_list[_current_dataset_idx].fetch();
    //_dataset_list[_current_dataset_idx].reset((decltype(result.get())*)NULL);
    return result;
}  

void DatasetContainer::async_download_data() {
    while (true) {
            //do download
        sleep(30);
    }
}

}//namespace feed
}//namespace custom_trainer
}//namespace paddle
