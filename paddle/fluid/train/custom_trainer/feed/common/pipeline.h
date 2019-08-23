#pragma once
#include <thread>
#include "paddle/fluid/framework/archive.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class DoneGuard {
public:
    DoneGuard(std::function<void()> func) : _func(func) {}
    virtual ~DoneGuard() { _func(); }
private:
    std::function<void()>  _func;
};

class PipelineOptions {
public:
    PipelineOptions() = default;
    uint32_t batch_size        = 10;        // pipe输出的batch大小
    uint32_t thread_num        = 1;         // converter的并发线程数
    float input_output_rate    = 1;         // 输入/输出 qps流量比
    uint32_t buffer_batch_count    = 4;     // pipe预存count组batch数据
    bool need_hold_input_data      = false; // 是否保存input流数据，否则消费后释放
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
    typedef std::function<int(TypeIn*, size_t in_num,
        TypeOut*, size_t* out_num, size_t thread_idx)> PipeDataConverter; 
    
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
        _output_channel->SetBlockSize(options.batch_size);
        size_t input_batch_size = options.batch_size * options.input_output_rate;
        _input_channel->SetBlockSize(input_batch_size);
        _input_data_buffer.resize(input_batch_size * options.buffer_batch_count);
        _output_data_buffer.resize(options.batch_size * options.buffer_batch_count);
        _output_channel->SetCapacity(_output_data_buffer.size());
        CHECK(_input_channel != nullptr) << " Input Channel is null";
        _convert_thread = std::make_shared<std::thread>([this](){
            async_convert_data();
        });
        return 0;
    }

    template <class PreTypeIn>
    int connect_to(Pipeline<PreTypeIn, TypeIn>& pre_pipeline, 
        PipelineOptions& options, PipeDataConverter data_converter) {
        // 保证全局batch一致
        options.batch_size = pre_pipeline.options().batch_size / options.input_output_rate;
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

    // 返回对input_channel的消费备份
    inline ::paddle::framework::Channel<TypeIn> backup_channel() {
        return _input_channel_backup;
    }
private:
    void async_convert_data() {
        size_t input_batch_size = _options.batch_size * _options.input_output_rate;
        while (!_is_read_end) {
            while (_output_channel->Size() < _input_data_buffer.size()) {
                size_t read_size = _input_channel->
                    Read(input_batch_size, &_input_data_buffer[0]);
                if (read_size == 0) {
                    _is_read_end = true;
                    break;
                }
                size_t write_size = 0;
                CHECK(_converter(&_input_data_buffer[0], read_size,
                    &_output_data_buffer[0], &write_size, 0) == 0) << "Data Converter Do Failed";
                _output_channel->WriteMove(write_size, &_output_data_buffer[0]);
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
