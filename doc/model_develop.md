# 如何添加自定义模型

## 动态图模型

Tips1: 必须在模型目录实现dygraph_model.py中的class DygraphModel，不能更改py文件名也不能更改class类名。

Tips2: 必须实现方法create_model, create_optimizer, create_metrics, train_forward, infer_forward。

Tips3: create_feeds和create_loss由train_forward和infer_forward内部调用，可以自定义方法名称。

### create_model

返回模型的class, 一般是调用net.py中定义的组网。

### create_feeds

解析batch_data, 返回paddle的tensor格式，在dataloader中yield是一条数据，注意这里返回的是Batch数据。

Tips: 因为动态图不需要占位符data, 这里实际返回的就是模型的输入tensor。

### create_loss

由于采用了动静一致的设计理念和方便计算指标的独立，将loss部分单独抽出来实现在这个函数中，也可以直接在train_forward中定义loss部分

### create_optimizer

定义优化器, 这里由用户自定义优化器。

### create_metrics

定义评估指标，返回打印的key值和声明的指标

Tips: 返回的指标必须是paddle.metric中的指标

### train_forward

自定义训练阶段，一般包含数据读入，计算loss损失，更新指标

Tips: 返回3个值，第一个必须是loss, 第二个是metric_list，可以为空list。第三个是想间隔打印的tensor dict, 可以返回None。

### infer_forward

除了不返回loss之外其他和train_forward相同，支持和train阶段不同的组网。



## 静态图模型

Tips1: 必须在模型目录实现static_model.py中的class StaticModel，不能更改py文件名也不能更改class类名。

Tips2: 必须实现方法create_feeds, net, infer_net, create_optimizer

### create_feeds

静态图采用graph结构，需要用paddle.static.data作为数据的占位符。

Tips1: 返回的feed_list的顺序必须和reader中yield的数据保持一致。

Tips2: 变长数据可以用lod_level=1表示，具体可参考models/rank/dnn/static_model_lod.py


### net

训练组网，请注意返回的是dict, 间隔打印。 key是打印的名称，value是对应的变量，

### infer_net

预测组网，如若和训练组网类似，可调用net部分。
