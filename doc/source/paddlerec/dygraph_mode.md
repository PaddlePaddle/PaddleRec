# 动态图训练

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
