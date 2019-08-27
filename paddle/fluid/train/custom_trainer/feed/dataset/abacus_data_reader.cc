#include "paddle/fluid/train/custom_trainer/feed/dataset/data_reader.h"

#include <cstdio>
#include <atomic>

#include <glog/logging.h>
#include <omp.h>

#include "paddle/fluid/train/custom_trainer/feed/io/file_system.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

/*解析Abacus格式明文Feasign
 */
class AbacusTextDataParser : public LineDataParser {
public:
    AbacusTextDataParser() {}
    virtual ~AbacusTextDataParser() {}

    virtual int parse_to_sample(const DataItem& data, SampleInstance& instance) const {
        instance.id = data.id;
        instance.labels.resize(1);
        size_t len = data.data.size();
        const char* str = data.data.c_str();
        const char* line_end = str + len;

        char* cursor = NULL;
        int show = (int)strtol(str, &cursor, 10);
        str = cursor;
        instance.labels[0] = (float)strtol(str, &cursor, 10);// click
        str = cursor;

        while (*(str += paddle::string::count_nonspaces(str)) != 0) {
            if (*str == '*') {
                str++;
                size_t len = paddle::string::count_nonspaces(str);
                str += len;
            } else if (*str == '$') {
                str++;
                CHECK(((int)strtol(str, &cursor, 10), cursor != str))<<" sample type parse err:" << str;
                str = cursor;
            } else if (*str == '#') {
                str++;
                break;
            } else if (*str == '@') {
                str++;
                size_t len = paddle::string::count_nonspaces(str);
                std::string all_str(str, str + len);
                str += len;
            } else {
                FeatureItem feature_item;
                feature_item.sign() = (uint64_t)strtoull(str, &cursor, 10);
                if (cursor == str) { //FIXME abacus没有这种情况
                    str++;
                    continue;
                }
                str = cursor;
                CHECK(*str++ == ':');
                CHECK(!isspace(*str));
                CHECK((feature_item.slot() = (int) strtol(str, &cursor, 10), cursor != str)) << " format error: " << str;
                str = cursor;
                instance.features.emplace_back(feature_item);
            }
        }
        VLOG(5) << "parse sample success, id:" << instance.id << ", fea_sum:" 
            << instance.features.size() << ", label:" << instance.labels[0];
        return 0;
    }
};
REGIST_CLASS(DataParser, AbacusTextDataParser);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
