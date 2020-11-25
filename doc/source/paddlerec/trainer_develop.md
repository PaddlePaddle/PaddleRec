# 如何添加自定义Trainer训练流程

模型训练的流程也可以像`model`及`reader`一样，由用户自定义，并在`config.yaml`中指定路径，由PaddleRec调用。

PaddleRec可自定义的流程有如下5个：
1. **instance**： 执行训练前的所有操作
2. **network**：执行组网的前向/反向，训练策略的添加
3. **startup**：执行模型的初始化，加载
4. **runnner**： 执行模型的训练
5. **terminal**： 执行训练后的所有操作

## instance

instance由GeneralTrainer首先调用，执行模型组网前的所有操作。用户可以在这里进行下载数据，import不同的包，配置环境变量等操作。instance的官方实现位于[instance.py](https://github.com/PaddlePaddle/PaddleRec/blob/master/core/trainers/framework/instance.py)，instance基类定义如下：

```python
class InstanceBase(object):
    def __init__(self, context):
        pass

    def instance(self, context):
        pass
```

您需要继承`InstanceBase`并命名为`Instance`，完成`instance`的实现，通过上下文信息字典`context`拿到模型所需信息，及保存相关配置。

## network

network将在instanc后调用，执行模型的组网。network的官方实现位于[network.py](https://github.com/PaddlePaddle/PaddleRec/blob/master/core/trainers/framework/network.py)，network基类定义如下：

```python
class NetworkBase(object):
    def __init__(self, context):
        pass

    def build_network(self, context):
        pass
```

可参照其他模式的实现方式，实现其中的部分步骤。您需要继承`NetworkBase`并命名为`Network`，完成`build_network`的实现，通过上下文信息字典`context`拿到模型所需信息，并在context中保存模型的program与scope信息，例如：

```python
context["model"][model_dict["name"]][
                "main_program"] = train_program
context["model"][model_dict["name"]][
    "startup_program"] = startup_program
context["model"][model_dict["name"]]["scope"] = scope
context["model"][model_dict["name"]]["model"] = model
context["model"][model_dict["name"]][
    "default_main_program"] = train_program.clone()
```

## startup

startup执行网络参数的初始化，或者模型的热启动，主要功能是执行`exe.run(fluid.default_startup_program())`。 startup的官方实现在[startup](https://github.com/PaddlePaddle/PaddleRec/blob/master/core/trainers/framework/startup.py)

```python
class StartupBase(object):
    def __init__(self, context):
        pass

    def startup(self, context):
        pass

    def load(self, context, is_fleet=False, main_program=None):
        dirname = envs.get_global_env(
            "runner." + context["runner_name"] + ".init_model_path", None)
        if dirname is None or dirname == "":
            return
        print("going to load ", dirname)
        if is_fleet:
            context["fleet"].load_persistables(context["exe"], dirname)
        else:
            fluid.io.load_persistables(
                context["exe"], dirname, main_program=main_program)
```

自定义startup流程，您需要继承`StartupBase`并命名为`Startup`，实现该类型中startup成员函数。

## runner

runner是运行的主要流程，主要功能是reader的运行，网络的运行，指标的打印以及模型的保存。以参数服务器Runner为示例，如下：

```python
class PSRunner(RunnerBase):
    def __init__(self, context):
        print("Running PSRunner.")
        pass

    def run(self, context):
        # 通过超参拿到迭代次数
        epochs = int(
            envs.get_global_env("runner." + context["runner_name"] +
                                ".epochs"))
        # 取第一个phase的模型与reader
        model_dict = context["env"]["phase"][0]
        for epoch in range(epochs):
            begin_time = time.time()
            # 调用run进行训练
            self._run(context, model_dict)
            end_time = time.time()
            seconds = end_time - begin_time
            print("epoch {} done, use time: {}".format(epoch, seconds))
            with fluid.scope_guard(context["model"][model_dict["name"]][
                    "scope"]):
                train_prog = context["model"][model_dict["name"]][
                    "main_program"]
                startup_prog = context["model"][model_dict["name"]][
                    "startup_program"]
                with fluid.program_guard(train_prog, startup_prog):
                    # 保存模型
                    self.save(epoch, context, True)
        context["status"] = "terminal_pass"
```

自定义runner需要参照官方实现[runner.py](https://github.com/PaddlePaddle/PaddleRec/blob/master/core/trainers/framework/runner.py)，继承基类`RunnerBase`，命名为`Runner`，并实现`run`成员函数。

## terminal

terminal主要进行分布式训练结束后的`stop worker`，以及其他需要在模型训练完成后进行的工作，比如数据整理，模型上传等等。

```python
class TerminalBase(object):
    def __init__(self, context):
        pass

    def terminal(self, context):
        print("PaddleRec Finish")
```

自定义terminal需要继承`TerminalBase`命名为`Terminal`，并实现成员函数`terminal`。

## 自定义流程参与训练

假如我们自定义了某个流程，将其与model/reader一样，放在workspace下，并同时更改yaml配置中的runner相关选项，PaddleRec会自动用指定的流程替换原始的类别。

```yaml
runner:
  - name: train_runner
    class: train
    epochs: 2
    device: cpu
    instance_class_path: "{workspace}/your_instance.py"
    network_class_path: "{workspace}/your_network.py"
    startup_class_path: "{workspace}/your_startup.py"
    runner_class_path: "{workspace}/your_runner.py"
    terminal_class_path: "{workspace}/your_terminal.py"
    print_interval: 1
```

## 示例

官方模型中的TDM是一个很好的示例，该模型自定义了`startup`的实现，可以参考[tdm_startup.py](https://github.com/PaddlePaddle/PaddleRec/blob/master/models/treebased/tdm/tdm_startup.py)

```python
class Startup(StartupBase):
    def startup(self, context):
        logger.info("Run TDM Trainer Startup Pass")
        if context["engine"] == EngineMode.SINGLE:
            self._single_startup(context)
        else:
            self._cluster_startup(context)

        context['status'] = 'train_pass'

    def _single_startup(self, context):
        # single process

    def _cluster_startup(self, context):
        # cluster process
```

于此同时，在yaml中更改了默认的startup执行类：
```yaml
runner:
- name: runner1
  class: train
  startup_class_path: "{workspace}/tdm_startup.py"
  epochs: 10
  device: cpu
```
