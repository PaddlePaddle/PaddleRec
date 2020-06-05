# PaddleRec 离线预测

## 单机离线预测启动配置

下面我们开始定义一个单机预测的`runner`:

```yaml
mode: runner_infer # 执行名为 runner1 的运行器

runner:
- name: runner_infer # 定义 runner 名为 runner1
  class: single_infer # 执行单机预测 class = single_infer
  device: cpu # 执行在 cpu 上
  init_model_path: "init_model" # 指定初始化模型的地址
  print_interval: 10 # 预测信息的打印间隔，以batch为单位
```

再定义具体的执行内容：

```yaml
phase:
- name: phase_infer # 该阶段名为 phase_infer
  model: "{workspace}/model.py" # 模型文件为workspace下的model.py
  dataset_name: dataset_infer # reader的名字

dataset:
- name: dataset_infer
  type: DataLoader # 使用DataLoader的数据读取方式
  batch_size: 2
  data_path: "{workspace}/test_data" # 数据地址
  sparse_slots: "click 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26" # sparse 输入的位置定义
  dense_slots: "dense_var:13"  # dense参数的维度定义

```