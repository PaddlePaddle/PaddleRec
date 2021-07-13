# 动态图模式介绍

Paddle2.0的特色之一是发布全新的动态图能力, PaddleRec推荐模型库也同时支持了动态图模式

## 即时反馈，快速调研

相较于之前静态图的方式，动态图更加灵活和轻量化，支持Python自带的Print打印输出方式，方便用户调试和快速调研

## 快速使用动态图

训练和预测的相关配置在每个模型文件中的config.yaml中配置，详细的yaml配置说明请参考进阶教程。

### 动态图训练

支持在任意目录下运行, 以下命令默认在PaddleRec根目录中运行

```bash
python -u tools/trainer.py -m models/rank/dnn/config.yaml
```
动态图训练的相关代码在tools/trainer.py，二次开发者可以在这个文件中快速开发。

### 动态图预测

```bash
python -u tools/infer.py -m models/rank/dnn/config.yaml
```
## 命令行修改配置

支持在命令行启动时，使用命令中的“-o”参数调整yaml文件中的配置。若使用命令传参与yaml文件指定参数两种方式对同一个参数赋予不同的值，则命令行传参的优先级更高，会覆盖yaml文件中指定的值。  
以dnn模型的use_gpu参数为例：
```bash
# 没有额外使用"-o"参数配置的情况下，按照config.yaml文件的配置，use_gpu参数值为false，将使用cpu运行。
# 使用"-o"参数指定config.yaml文件中use_gpu的值为true，因为命令行传参的优先级更高，将使用gpu运行
python -u tools/trainer.py -m models/rank/dnn/config.yaml -o runner.use_gpu=true
python -u tools/infer.py -m models/rank/dnn/config.yaml -o runner.use_gpu=true
```
