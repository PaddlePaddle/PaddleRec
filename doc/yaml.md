```
# 全局配置
debug: false
workspace: "."


# 用户可以配多个dataset，exector里不同阶段可以用不同的dataset
dataset:
  - name: sample_1
    type: DataLoader #或者QueueDataset
    batch_size: 5
    data_path: "{workspace}/data/train"
    # 用户自定义reader
    data_converter: "{workspace}/rsc15_reader.py"

  - name: sample_2
    type: QueueDataset #或者DataLoader
    batch_size: 5
    data_path: "{workspace}/data/train"
    # 用户可以配置sparse_slots和dense_slots，无需再定义data_converter
    sparse_slots: "click ins_weight 6001 6002 6003 6005 6006 6007 6008 6009"
    dense_slots: "readlist:9"


#示例一，用户自定义参数，用于组网配置
hyper_parameters:
    #优化器
    optimizer：
        class: Adam
        learning_rate: 0.001
        strategy: "{workspace}/conf/config_fleet.py"
    # 用户自定义配置
    vocab_size: 1000
    hid_size: 100
    my_key1: 233
    my_key2: 0.1


mode: runner1

runner:
  - name: runner1 # 示例一，train
    trainer_class: single_train
    epochs: 10
    device: cpu
    init_model_path: ""
    save_checkpoint_interval: 2
    save_inference_interval: 4
    # 下面是保存模型路径配置
    save_checkpoint_path: "xxxx"
    save_inference_path: "xxxx"

  - name: runner2 # 示例二，infer
    trainer_class: single_train
    epochs: 1
    device: cpu
    init_model_path: "afs:/xxx/xxx"



phase:
- name: phase1
  model: "{workspace}/model.py"
  dataset_name: sample_1
  thread_num: 1
```
