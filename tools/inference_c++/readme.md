# C++端预测部署
paddlerec目前提供C++的预测部署方案，用户通过paddlerec训练的模型使用save_inference_model接口保存后，即可根据本教程开发c++端预测部署。本教程将以wide_deep模型为例，详细说明如何使用。  
如果您刚刚接触Paddle Inference不久，建议您[访问这里](https://paddle-inference.readthedocs.io/en/latest/#)对Paddle Inference做一个初步的认识。

## 环境准备
os：linux  
gcc：8.2  
CMake 3.0+  
cuda：10.2（仅在使用GPU版本的预测库时需要）  

推荐使用docker配置环境：  
docker pull registry.baidubce.com/paddlepaddle/serving:latest-cuda10.2-cudnn8-devel  
docker中需自行安装[paddle](https://www.paddlepaddle.org.cn/)  
sudo apt install cmake

## 前置条件
### step1：使用save_inference_model接口保存下来的模型
将使用save_inference_model接口保存下来的模型放在单独的目录中，以备读取。在本例中，我们准备了wide_deep模型放在tools/inference_c++/目录中。可以访问[这里](https://paddlerec.bj.bcebos.com/wide_deep/wide_deep.tar)获取模型并自行解压。目录结构为：
```
wide_deep
├── rec_inference.pdiparams
└── rec_inference.pdmodel
```

### step2：下载PaddlePaddle C++ 预测库 fluid_inference
PaddlePaddle C++ 预测库针对不同的`CPU`和`CUDA`版本提供了不同的预编译版本，请根据实际情况下载:  [C++预测库下载列表](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html)
下载后请在tools/inference_c++/目录下解压，目录结构为：
```
paddle_inference
├── CMakeCache.txt 
├── paddle # paddle核心库和头文件
|
├── third_party # 第三方依赖库和头文件
|
└── version.txt # 版本和编译信息
```

### step3: 为您的模型开发一套reader
在本教程中，我们以wide_deep模型为例，编写了文件`criteo_reader.h`。若您希望更换为自己的模型，需要按照以下要求自行编写reader，并加入`inference.cpp`文件的include中。
1. reader的主体为一个`void FeederProcess(std::string path, SharedQueue<click_joint> &queue)`函数。需要以`FeederProcess`为函数名，输入参数为数据路径和存储数据的队列。在函数中，您需要根据数据路径获取数据文件，并按行读取其中的数据。将每一条数据放入queue队列的一个元素中，这里需要注意每个slot的名字需命名为模型中保存的input_name。
2. reader中需要定义`struct click_joint `的结构体。每个结构体为queue队列的一个元素。在预测时会先根据name_type确定每个input_name的类型为int或float。再根据input_name从相应类型的vector中获取数据。
3. 您的reader中需要加入`extern std::atomic<int> feeder_stops;`标志reader进程是否完结，并在`FeederProcess`函数退出时添加`feeder_stops.fetch_add(1);`标志完结。在预测时，程序会根据此判断所有reader线程是否全部退出不再输入数据。

## 编译并运行
我们准备了run.sh脚本提供一键编译并运行。  
运行`bash run.sh`，会在目录下产生build目录，目录中的inference文件即为可执行文件。  
启动的参数如下：
|        名称         |    类型    |             取值             | 是否必须 |                               作用描述                               |
| :-----------------: | :-------: | :--------------------------: | :-----: | :------------------------------------------------------------------: |
|       --model_file        |    string    |       任意路径         |    是    |                            模型文件路径（当需要从磁盘加载 Combined 模型时使用）                           |
|       --params_file        |    string    |       任意路径         |    是    |                            参数文件路径 （当需要从磁盘加载 Combined 模型时使用）                           |
|       --model_dir        |    string    |       任意路径         |    是    |                            模型文件夹路径 （当需要从磁盘加载非 Combined 模型时使用）                           |
|       --use_gpu        |    bool    |       True/False         |    是    |                            是否使用gpu                            |
|       --data_dir        |    string    |       任意路径         |    是    |                            测试数据目录                            |
|       --batchsize        |    int    |       >= 1         |    是    |                            批训练样本数量                            |
