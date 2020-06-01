```yaml
# 全局配置
# Debug 模式开关，Debug模式下，会打印OP的耗时及IO占比
debug: false

# 工作区目录
# 使用文件夹路径，则会在该目录下寻找超参配置，组网，数据等必须文件
workspace: "/home/demo_model/"
# 若 workspace: paddlerec.models.rank.dnn
# 则会使用官方默认配置与组网


# 用户可以指定多个dataset(数据读取配置)
# 运行的不同阶段可以使用不同的dataset
dataset:
  # dataloader 示例
  - name: dataset_1
    type: DataLoader 
    batch_size: 5
    data_path: "{workspace}/data/train"
    # 指定自定义的reader.py所在路径
    data_converter: "{workspace}/rsc15_reader.py"

  # QueueDataset 示例
  - name: dataset_2
    type: QueueDataset 
    batch_size: 5
    data_path: "{workspace}/data/train"
    # 用户可以配置sparse_slots和dense_slots，无需再定义data_converter，使用默认reader
    sparse_slots: "click ins_weight 6001 6002 6003 6005 6006 6007 6008 6009"
    dense_slots: "readlist:9"


# 自定义超参数，主要涉及网络中的模型超参及优化器
hyper_parameters:
    #优化器
    optimizer:
      class: Adam # 直接配置Optimizer，目前支持sgd/Adam/AdaGrad
      learning_rate: 0.001
      strategy: "{workspace}/conf/config_fleet.py" # 使用大规模稀疏pslib模式的特有配置
    # 模型超参
    vocab_size: 1000
    hid_size: 100


# 通过全局参数mode指定当前运行的runner
mode: runner_1

# runner主要涉及模型的执行环境，如：单机/分布式，CPU/GPU，迭代轮次，模型加载与保存地址
runner:
  - name: runner_1 # 配置一个runner，进行单机的训练
    class: single_train # 配置运行模式的选择，还可以选择：single_infer/local_cluster_train/cluster_train
    epochs: 10
    device: cpu
    init_model_path: ""
    save_checkpoint_interval: 2
    save_inference_interval: 4
    # 下面是保存模型路径配置
    save_checkpoint_path: "xxxx"
    save_inference_path: "xxxx"

  - name: runner_2 # 配置一个runner，进行单机的预测
    class: single_infer
    epochs: 1
    device: cpu
    init_model_path: "afs:/xxx/xxx"


# 模型在训练时，可能存在多个阶段，每个阶段的组网与数据读取都可能不尽相同
# 每个runner都会完整的运行所有阶段
# phase指定运行时加载的模型及reader
phase:
- name: phase1
  model: "{workspace}/model.py"
  dataset_name: sample_1
  thread_num: 1
```
