### Avazu dataset for FLEN models:
#### 1.Get raw datasets:
you can go to：[https://www.kaggle.com/c/avazu-ctr-prediction/data](https://www.kaggle.com/c/avazu-ctr-prediction)

将下载的原始数据目录配置在data_config.yaml中，执行命令获取全量数据

| raw_file_dir | 原始数据集目录 | 
| -------- | -------- | 
| raw_filled_file_dir     | 原始数据缺失值处理后的目录     |
|   train_data_dir   | 训练集存放目录     | 
|   test_data_dir   | 测试集存放目录     | 
| rebuild_feature_map     | 是否重建类别特征，默认为True     |
| min_threshold     | 类别特征计数临界值，默认为4    |
| feature_map_cache     | 特征缓存数据     | 



```bash
sh data_process.sh
```
#### 2.Get preprocessd datasets:
you can also go to: [AiStudio数据集](https://aistudio.baidu.com/aistudio/datasetdetail/125200)

