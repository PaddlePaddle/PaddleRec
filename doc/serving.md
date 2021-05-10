# 在线Serving部署
PaddleRec训练出来的模型可以使用[Serving](https://github.com/PaddlePaddle/Serving)部署在服务端。  
本教程以[wide_deep](../models/rank/wide_deep/README.md)模型使用demo数据为例进行部署  

## 首先使用save_inference_model接口保存模型
1. 首先需要在模型的yaml配置中，加入use_inference参数，并把值设置成True。use_inference决定是否使用save_inference_model接口保存模型，默认为否。  
2. 确定需要的输入和输出的预测模型变量，将其变量名以字符串的形式填入save_inference_feed_varnames和save_inference_fetch_varnames列表中。  
以wide_deep模型为例，可以在其config.yaml文件中观察到如下结构。训练及测试数据集选用[Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/)所用的Criteo数据集。该数据集包括两部分：训练集和测试集。训练集包含一段时间内Criteo的部分流量，测试集则对应训练数据后一天的广告点击流量。feed参数的名字中```<label>```表示广告是否被点击，点击用1表示，未点击用0表示，```<integer feature>```代表数值特征（连续特征dense_input），共有13个连续特征，```<categorical feature>```代表分类特征（离散特征C1~C26），共有26个离散特征。fetch参数输出的是auc，具体意义为static_model.py里def net（）函数中将auc使用cast转换为float32类型语句中的cast算子。  
```yaml
runner:
  # 通用配置不再赘述
  ...
  # use inference save model
  use_inference: True  # 静态图训练时保存为inference model
  save_inference_feed_varnames: ["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14","C15","C16","C17","C18","C19","C20","C21","C22","C23","C24","C25","C26","dense_input"] # inference model 的feed参数的名字
  save_inference_fetch_varnames: ["sigmoid_0.tmp_0"] # inference model 的fetch参数的名字
```
3. 启动静态图训练
```bash
# 进入模型目录
# cd models/rank/wide_deep # 在任意目录均可运行
# 静态图训练
python -u ../../../tools/static_trainer.py -m config.yaml # 全量数据运行config_bigdata.yaml 
```

## 安装 paddle serving
强烈建议您在我们提供的Docker内构建Paddle Serving，请查看[如何在Docker中运行PaddleServing](https://github.com/PaddlePaddle/Serving/blob/develop/doc/RUN_IN_DOCKER_CN.md)
```bash
# 安装 paddle-serving-client
pip install paddle-serving-client -i https://mirror.baidu.com/pypi/simple

# 安装 paddle-serving-server
pip install paddle-serving-server -i https://mirror.baidu.com/pypi/simple

# 安装 paddle-serving-server-gpu
pip install paddle-serving-server-gpu -i https://mirror.baidu.com/pypi/simple
```

## 导出模型
您可以使用Paddle Serving提供的名为`paddle_serving_client.convert`的内置模块进行转换。
```bash
python -m paddle_serving_client.convert --dirname ./your_inference_model_dir --model_filename ./your_inference_model_filename --params_filename ./your_inference_params_filename
```

也可以通过Paddle Serving的`inference_model_to_serving`接口转换成可用于Paddle Serving的模型文件。
```python
import paddle_serving_client.io as serving_io
serving_io.inference_model_to_serving(dirname, serving_server="serving_server", serving_client="serving_client",  model_filename=None, params_filename=None)
```

模块参数与`inference_model_to_serving`接口参数相同。
| 参数 | 类型 | 默认值 | 描述 |
|--------------|------|-----------|--------------------------------|
| `dirname` | str | - | 需要转换的模型文件存储路径，Program结构文件和参数文件均保存在此目录。|
| `serving_server` | str | `"serving_server"` | 转换后的模型文件和配置文件的存储路径。默认值为serving_server |
| `serving_client` | str | `"serving_client"` | 转换后的客户端配置文件存储路径。默认值为serving_client |
| `model_filename` | str | None | 存储需要转换的模型Inference Program结构的文件名称。如果设置为None，则使用 `__model__` 作为默认的文件名 |
| `params_filename` | str | None | 存储需要转换的模型所有参数的文件名称。当且仅当所有模型参数被保>存在一个单独的二进制文件中，它才需要被指定。如果模型参数是存储在各自分离的文件中，设置它的值为None |

以上命令会生成serving_client和serving_server两个文件夹
```txt
├── serving_client
    ├── serving_client_conf.prototxt # 模型输入输出信息
    ├── serving_client_conf.stream.prototxt
├── serving_server
    ├── __model__
    ├── __params__
    ├── serving_server_conf.prototxt
    ├── serving_server_conf.stream.prototxt
```

## 启动PaddleServing服务
服务端我们提供rpc和web两种方式，您可以选择一种启动。
### 启动rpc服务端
```bash
# GPU
python -m paddle_serving_server_gpu.serve --model serving_server --port 9393 --gpu_ids 0

# CPU
python -m paddle_serving_server.serve --model serving_server --port 9393
```

| 参数 | 类型 | 默认值 | 描述 |
|--------------|------|-----------|--------------------------------|
| `thread` | int | `4` | Concurrency of current service |
| `port` | int | `9292` | Exposed port of current service to users|
| `name` | str | `""` | Service name, can be used to generate HTTP request url |
| `model` | str | `""` | Path of paddle model directory to be served |
| `mem_optim_off` | - | - | Disable memory optimization |
| `ir_optim` | - | - | Enable analysis and optimization of calculation graph |
| `use_mkl` (Only for cpu version) | - | - | Run inference with MKL |
| `use_trt` (Only for Cuda>=10.1 version) | - | - | Run inference with TensorRT  |
| `use_lite` (Only for ARM) | - | - | Run PaddleLite inference |
| `use_xpu` (Only for ARM+XPU) | - | - | Run PaddleLite XPU inference |

### 启动web服务端
运行PaddleRec/tools目录下的webserer.py文件，传入两个参数，第一个参数指定使用设备为cpu还是gpu，第二个参数指定端口号。
```bash
# GPU
python ../../../tools/webserver.py gpu 9393

# CPU
python ../../../tools/webserver.py cpu 9393
```

## 测试部署的服务
在服务器端启动serving服务成功后，部署客户端需要您打开新的终端页面。
```bash
# 进入模型目录
# cd models/rank/wide_deep # 在任意目录均可运行
# 启动客户端
python -u ../../../tools/rec_client.py --client_config=serving_client/serving_client_conf.prototxt --connect=0.0.0.0:9393 --use_gpu=true --data_dir=data/sample_data/train/ --reader_file=criteo_reader.py --batchsize=5 --client_mode=web
```

| 参数 | 类型 | 默认值 | 描述 |
|--------------|------|-----------|--------------------------------|
| `client_config` | str | - | [导出模型](#导出模型)步骤中生成的serving_client目录下的prototxt文件 |
| `connect` | str | - | 服务进程指定的ip以及端口 |
| `use_gpu` | bool | - | 是否使用gpu |
| `data_dir` | str | - | 数据集所在的目录 |
| `reader_file` | str | - | 模型指定的reader，能够将数据读取进来按行完成预处理 |
| `batchsize` | int | - | 数据的batch_size大小 |
| `client_mode` | str | - | 使用rpc方式或web方式，取值为“rpc”或“web” |
