# 在线部署

PaddleRec以飞桨框架为底层，因此训练保存出的inference_model(预测模型)，可以使用飞桨强大的部署能力快速在线部署。

首先我们在`yaml`配置中，指定inference_model的保存间隔与保存地址：

```yaml
mode: runner_train # 执行名为 runner_train 的运行器

runner:
- name: runner_train # 定义 runner 名为 runner_train
  class: single_train # 执行单机训练 class = single_train
  device: cpu # 执行在 cpu 上
  epochs: 10 # 训练轮数

  save_checkpoint_interval: 2 # 每隔2轮保存一次checkpoint
  save_inference_interval: 4 # 每个4轮保存依次inference model
  save_checkpoint_path: "increment" # checkpoint 的保存地址
  save_inference_path: "inference" # inference model 的保存地址
  save_inference_feed_varnames: [] # inference model 的feed参数的名字
  save_inference_fetch_varnames: [] # inference model 的fetch参数的名字
  init_model_path: "" # 如果是加载模型热启，则可以指定初始化模型的地址
  print_interval: 10 # 训练信息的打印间隔，以batch为单位
```

训练完成后，我们便可以在`inference`或`increment`文件夹中看到保存的模型/参数。

参考以下链接进行模型的不同场景下的部署。

### [服务器端部署](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/index_cn.html)

### [移动端部署](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/mobile/index_cn.html)

### [在线Serving](https://github.com/PaddlePaddle/Serving)

### [模型压缩](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/paddleslim/paddle_slim.html)
