# 静态图训练

静态图相比动态图性能更好，分布式训练目前只支持静态图模式

## 快速使用静态图

训练和预测的相关配置在每个模型文件中的config.yaml中配置，详细的yaml配置说明请参考进阶教程。

### 静态图训练

支持在任意目录下运行, 以下命令默认在PaddleRec根目录中运行

```bash
python -u tools/static_trainer.py -m models/rank/dnn/config.yaml
```
静态图训练的相关代码在tools/static_trainer.py，二次开发者可以在这个文件中快速开发。

### 静态图预测

```bash
python -u tools/static_infer.py -m models/rank/dnn/config.yaml
```
