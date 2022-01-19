# 可视化功能介绍
PaddleRec通过飞桨生态的可视化分析工具VisualDL，支持将训练的过程可视化，让您清晰而直观的看到模型的训练效果。

## 可视化功能的依赖
可视化功能依赖飞桨生态的可视化分析工具VisualDL完成，如果需要开启这项功能需要先安装VisualDL。安装命令如下：
```bash
python -m pip install visualdl -i https://mirror.baidu.com/pypi/simple
```

## 开启可视化功能
1. 在各模型的yaml配置文件中，runner项下添加新的参数“use_visual”，并将该项的值填写为True。该参数为bool类型，默认值为False，用于在安装VisualDL完成的情况下开启可视化训练。
2. 在模型的dygraph_model.py文件中，可以通过train_forward函数的metrics_list, print_dict两个返回值来输出动态图运行时您需要打印的指标或变量。同理在模型的static_model.py文件中，可以通过net函数的fetch_dict返回值来输出静态图运行时您需要打印的指标。可视化功能会自动收集这些指标，并创建一个visualDL_log目录存放他们。
3. 您可以正常的训练模型
4. 启动VisualDL面板，有一下两种方法供您选择：

使用命令行启动VisualDL面板，命令格式如下：
```python
visualdl --logdir <dir_1, dir_2, ... , dir_n> --model <model_file> --host <host> --port <port> --cache-timeout <cache_timeout> --language <language> --public-path <public_path> --api-only
```

参数详情：

| 参数            | 意义                                                         |
| --------------- | ------------------------------------------------------------ |
| --logdir        | 设定日志所在目录，可以指定多个目录，VisualDL将遍历并且迭代寻找指定目录的子目录，将所有实验结果进行可视化 |
| --model         | 设定模型文件路径(非文件夹路径)，VisualDL将在此路径指定的模型文件进行可视化，目前可支持PaddlePaddle、ONNX、Keras、Core ML、Caffe等多种模型结构，详情可查看[graph支持模型种类](https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/components/README_CN.md#%E5%8A%9F%E8%83%BD%E6%93%8D%E4%BD%9C%E8%AF%B4%E6%98%8E-4) |
| --host          | 设定IP，默认为`127.0.0.1`，若想使得本机以外的机器访问启动的VisualDL面板，需指定此项为`0.0.0.0`或自己的公网IP地址                                    |
| --port          | 设定端口，默认为`8040`                                       |
| --cache-timeout | 后端缓存时间，在缓存时间内前端多次请求同一url，返回的数据从缓存中获取，默认为20秒 |
| --language      | VisualDL面板语言，可指定为'en'或'zh'，默认为浏览器使用语言   |
| --public-path   | VisualDL面板URL路径，默认是'/app'，即访问地址为'http://&lt;host&gt;:&lt;port&gt;/app' |
| --api-only      | 是否只提供API，如果设置此参数，则VisualDL不提供页面展示，只提供API服务，此时API地址为'http://&lt;host&gt;:&lt;port&gt;/&lt;public_path&gt;/api'；若没有设置public_path参数，则默认为'http://&lt;host&gt;:&lt;port&gt;/api' |

使用Python脚本启动VisualDL面板，接口如下：

```python
visualdl.server.app.run(logdir,
                        model="path/to/model",
                        host="127.0.0.1",
                        port=8080,
                        cache_timeout=20,
                        language=None,
                        public_path=None,
                        api_only=False,
                        open_browser=False)
```

请注意：除`logdir`外，其他参数均为不定参数，传递时请指明参数名。

接口参数具体如下：

| 参数          | 格式                                             | 含义                                                         |
| ------------- | ------------------------------------------------ | ------------------------------------------------------------ |
| logdir        | string或list[string_1, string_2, ... , string_n] | 日志文件所在的路径，VisualDL将在此路径下递归搜索日志文件并进行可视化，可指定单个或多个路径，每个路径中及其子目录中的日志都将视为独立日志展现在前端面板上 |
| model         | string                                           | 模型文件路径(非文件夹路径)，VisualDL将在此路径指定的模型文件进行可视化，目前可支持PaddlePaddle、ONNX、Keras、Core ML、Caffe等多种模型结构，详情可查看[graph支持模型种类](https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/components/README_CN.md#%E5%8A%9F%E8%83%BD%E6%93%8D%E4%BD%9C%E8%AF%B4%E6%98%8E-4) |
| host          | string                                           | 设定IP，默认为`127.0.0.1`，若想使得本机以外的机器访问启动的VisualDL面板，需指定此项为`0.0.0.0`或自己的公网IP地址                       |
| port          | int                                              | 启动服务端口，默认为`8040`                                   |
| cache_timeout | int                                              | 后端缓存时间，在缓存时间内前端多次请求同一url，返回的数据从缓存中获取，默认为20秒 |
| language      | string                                           | VisualDL面板语言，可指定为'en'或'zh'，默认为浏览器使用语言   |
| public_path   | string                                           | VisualDL面板URL路径，默认是'/app'，即访问地址为'http://&lt;host&gt;:&lt;port&gt;/app' |
| api_only      | boolean                                          | 是否只提供API，如果设置此参数，则VisualDL不提供页面展示，只提供API服务，此时API地址为'http://&lt;host&gt;:&lt;port&gt;/&lt;public_path&gt;/api'；若没有设置public_path参数，则默认为'http://&lt;host&gt;:&lt;port&gt;/api' |
| open_browser  | boolean                                          | 是否打开浏览器，设置为True则在启动后自动打开浏览器并访问VisualDL面板，若设置api_only，则忽略此参数 |

5. 在使用任意一种方式启动VisualDL面板后，打开浏览器访问VisualDL面板，即可查看日志的可视化结果

## 注意：
1. 可视化功能依赖visualDL实现，请先安装最新版visualDL再开启yaml文件中的use_visual功能，不然会报错。
2. 目前我们不支持静态图中dataset方式的可视化
3. 目前可视化功能仅支持生成折线图，后续会逐步添加更多功能的可视化，敬请期待。
4. 若对功能有疑问欢迎来用户群中交流：QQ群号码：861717190，微信小助手微信号：paddlerec2020
