## 数据准备
### 原始数据获取
1. mind https://msnews.github.io/index.html
2. glove.840B.300d  https://nlp.stanford.edu/projects/glove/
3. KG file： 给作者发邮件taoqi.qt@gmail.com
### 直接下载处理好的数据和模型权重
https://aistudio.baidu.com/aistudio/datasetdetail/80869

## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。  
在全量数据下模型的指标如下：  

| 模型  | AUC |  MRR   |    nDCG5 |   nDCG10  | batch_size | epoch_num | Time of each epoch |
|-----|-----|-----|-----|-----|------------|-----------|--------------------|
| kim |  0.6681   |   0.3164  |    0.3484 |  0.4132   | 16         | 7         | 2h                 |
|  kim   |   0.6696  |   0.3192  |   0.3515  |  0.4158   | 16         | 8         | 2h                 |

```bash
python -u trainer.py -m config_bigdata.yml -o mode=train
python -u infer.py -m config_bigdata.yml -o mode=test
```
**因为训练评估比较耗时建议** ```sh run.sh```
