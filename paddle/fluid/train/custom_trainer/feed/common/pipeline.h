#pragma once
#include "paddle/fluid/framework/archive.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class PipelineOptions {
public:
    PipelineOptions() = default;
    uint32_t buffer_data_num       = 400  ;  //缓冲区数据个数，需大于batch_size
    uint32_t batch_size            = 100  ;  //从pipe读数据的batch大小
    bool need_hold_input_data      = false;  //是否保存input流数据，否则消费后释放
};

/*
 *  数据流管道,管道内可对流入数据进行格式转换，再流出
 *  
 *  |---------------Pipeline---------------|
 *  Channel<IN> -> Converter -> Channel<OUT>
 *  多个管道可通过connect_to方法进行级联 
 * 
 *  使用initialize 或 connect_to 初始化管道
 */
template <class TypeIn, class TypeOut>
class Pipeline {
public: 
    Pipeline() {}
    Pipeline(Pipeline&&) = delete; 
    Pipeline(const Pipeline&) = delete; 
    typedef std::function<int(const TypeIn*, TypeOut*, size_t num)> PipeDataConverter; 
    
    int initialize(const PipelineOptions& options,
        ::paddle::framework::Channel<TypeIn> input_channel, 
        PipeDataConverter data_converter) {      
        CHECK(_inited == false);
        CHECK(options.batch_size > 0);
        _inited = true;
        _options = options;
        _is_read_end = false;
        _converter = data_converter;
        _input_channel = input_channel;
        _output_channel = ::paddle::framework::MakeChannel<TypeOut>();

        auto batch_size = options.batch_size;
        auto buffer_data_num = options.buffer_data_num;
        _input_channel->SetBlockSize(batch_size);
        _output_channel->SetBlockSize(batch_size);
        _input_data_buffer.resize(buffer_data_num);
        _output_data_buffer.resize(buffer_data_num);
        if (buffer_data_num / batch_size < 3) {
            buffer_data_num = batch_size * 3;
        }
        buffer_data_num = (buffer_data_num / batch_size) * batch_size;
        _output_channel->SetCapacity(buffer_data_num);
        CHECK(_input_channel != nullptr) << " Input Channel is null";
        _convert_thread = std::make_shared<std::thread>([this](){
            async_convert_data();
        });
        return 0;
    }

    template <class PreTypeIn>
    int connect_to(Pipeline<PreTypeIn, TypeIn>& pre_pipeline, 
        PipeDataConverter data_converter) {
        return initialize(pre_pipeline.options(), pre_pipeline.output_chnnel(), data_converter);
    }
    
    virtual ~Pipeline() {
        _is_read_end = true;
        if (_convert_thread != nullptr) {
            _convert_thread->join();
        }
    }

    inline size_t read(std::vector<TypeOut>& p) {
        p.clear();
        size_t num = _output_channel->Read(p);
        return num;
    }

    inline const PipelineOptions& options() {
        return _options;
    }

    inline ::paddle::framework::Channel<TypeOut> output_chnnel() {
        return _output_channel;
    }
private:
    void async_convert_data() {
        size_t convete_batch_size =  _input_data_buffer.size() / 4;
        if (convete_batch_size < _options.batch_size * 3) {
            convete_batch_size = 3 * _options.batch_size;
        }
        convete_batch_size = (convete_batch_size / _options.batch_size) * _options.batch_size;
        while (!_is_read_end) {
            while (_output_channel->Size() < _input_data_buffer.size()) {
                size_t read_size = _input_channel->
                    Read(convete_batch_size, &_input_data_buffer[0]);
                if (read_size == 0) {
                    _is_read_end = true;
                    break;
                }
                CHECK(_converter(&_input_data_buffer[0], &_output_data_buffer[0], 
                    read_size) == 0) << "Data Converter Do Failed";
                _output_channel->WriteMove(read_size, &_output_data_buffer[0]);
                if (_options.need_hold_input_data) {
                    _input_channel_backup->WriteMove(read_size, &_input_data_buffer[0]);
                }
            }  
            sleep(1);
        }
    }    
   
    
private:
    bool _inited      = false;                                     //标识初始化状态
    bool _is_read_end = false;                                     //标识输入流读取完成
    PipelineOptions _options;                                      //pipe参数
    PipeDataConverter _converter;                                  //converter
    std::vector<TypeIn> _input_data_buffer;                        //输入数据buffer
    std::vector<TypeOut> _output_data_buffer;                      //出数据buffer
    std::shared_ptr<std::thread> _convert_thread;                  //异步convert
    ::paddle::framework::Channel<TypeIn> _input_channel;           //输入流
    ::paddle::framework::Channel<TypeIn> _input_channel_backup;    //备份原始输入流
    ::paddle::framework::Channel<TypeOut> _output_channel;         //输出流
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
