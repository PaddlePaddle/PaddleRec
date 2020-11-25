# 如何给模型增加Metric

## PaddleRec Metric使用示例
```
from paddlerec.core.model import ModelBase
from paddlerec.core.metrics import RecallK

class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def net(self, inputs, is_infer=False):
        ...
        acc = RecallK(input=logits, label=label, k=20)
        self._metrics["Train_P@20"] = acc
```
## Metric类
### 成员变量
> _global_metric_state_vars（dict), 
字典类型，用以存储metric计算过程中需要的中间状态变量。一般情况下，这些中间状态需要是Persistable=True的变量，所以会在模型保存的时候也会被保存下来。因此infer阶段需手动将这些中间状态值清零，进而保证预测结果的正确性。

### 成员函数
> clear(self, scope):
从scope中将self._global_metric_state_vars中的状态值全清零。该函数一般用在**infer**阶段开始的时候。用以保证预测指标的正确性。

> calc_global_metrics(self, fleet, scope=None):
将self._global_metric_state_vars中的状态值在所有训练节点上做all_reduce操作，进而下一步调用_calculate()函数计算全局指标。若fleet=None，则all_reduce的结果为自己本身，即单机全局指标计算。

> get_result(self): 返回训练过程中需要fetch，并定期打印至屏幕的变量。返回类型为dict。

## Metrics
### AUC
```python
AUC(input ,label, curve='ROC', num_thresholds=2**12 - 1, topk=1, slide_steps=1)
```
Auc，全称Area Under the Curve(AUC)，该层根据前向输出和标签计算AUC，在二分类(binary classification)估计中广泛使用。在二分类(binary classification)中广泛使用。相关定义参考 https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve 。

#### 参数
- **input(Tensor|LoDTensor)**: 数据类型为float32，float64。浮点二维变量。输入为网络的预测值。shape为[batch_size, 2]。
- **label(Tensor|LoDTensor)**: 数据类型为int64，int32。输入为数据集的标签。shape为[batch_size, 1]。
- **curve(str)**: 曲线类型，可以为 ROC 或 PR，默认 ROC。 
- **num_thresholds(int)**: 将roc曲线离散化时使用的临界值数。默认200。
- **topk(int)**: 取topk的输出值用于计算。
- **slide_steps(int)**: - 当计算batch auc时，不仅用当前步也用于先前步。slide_steps=1，表示用当前步；slide_steps = 3表示用当前步和前两步；slide_steps = 0，则用所有步。

#### 返回值
该指标训练过程中定期的变量有两个：
- **AUC**: 整体AUC值
- **BATCH_AUC**：当前batch的AUC值


### PrecisionRecall
```python
PrecisionRecall(input, label, class_num)
```
计算precison, recall, f1。

#### 参数
- **input(Tensor|LoDTensor)**: 数据类型为float32,float64。输入为网络的预测值。shape为[batch_size, class_num]
- **label(Tensor|LoDTensor)**: 数据类型为int32。输入为数据集的标签。shape为 [batch_size, 1] 
- **class_num(int)**: 类别个数。

#### 返回值
- **[TP FP TN FN]**: 形状为[class_num, 4]的变量，用以表征每种类型的TP，FP，TN和FN值。TP=true positive, FP=false positive, TN=true negative, FN=false negative。若需计算每种类型的precison, recall，f1, 则可根据如下公式进行计算：
precision = TP / (TP + FP); recall = TP = TP / (TP + FN); F1 = 2 * precision * recall / (precision + recall)。

- **precision_recall_f1**: 形状为[6]，分别代表[macro_avg_precision, macro_avg_recall, macro_avg_f1, micro_avg_precision, micro_avg_recall, micro_avg_f1]，这里macro代表先计算每种类型的准确率，召回率，F1，然后求平均。micro代表先计算所有类型的整体TP，TN， FP, FN等中间值，然后在计算准确率，召回率，F1.


### RecallK
```python
RecallK(input, label, k=20)
```
TopK的召回准确率，对于任意一条样本来说，若前top_k个分类结果中包含正确分类标签，则视为正样本。

#### 参数
- **input(Tensor|LoDTensor)**: 数据类型为float32,float64。输入为网络的预测值。shape为[batch_size, class_dim]
- **label(Tensor|LoDTensor)**: 数据类型为int64，int32。输入为数据集的标签。shape为 [batch_size, 1] 
- **k(int)**: 取每个类别中top_k个预测值用于计算召回准确率。

#### 返回值
- **InsCnt**：样本总数
- **RecallCnt**: topk可以正确被召回的样本数
- **Acc(Recall@k)**: RecallCnt/InsCnt，即Topk召回准确率。

## PairWise_PN
```python
PosNegRatio(pos_score, neg_score)
```
正逆序指标，一般用在输入是pairwise的模型中。例如输入既包含正样本，也包含负样本，模型需要去学习最大化正负样本打分的差异。

#### 参数
- **pos_score(Tensor|LoDTensor）**: 正样本的打分，数据类型为float32，float64。浮点二维变量，值的范围为[0,1]。
- **neg_score(Tensor|LoDTensor)**：负样本的打分。数据类型为float32，float64。浮点二维变量，值的范围为[0,1]。

#### 返回值
- **RightCnt**: pos_score > neg_score的样本数
- **WrongCnt**: pos_score <= neg_score的样本数
- **PN**: (RightCnt + 1.0) / (WrongCnt + 1.0), 正逆序，+1.0是为了避免除0错误。

### Customized_Metric
如果你需要在自定义metric，那么你需要按如下步骤操作：
1. 继承paddlerec.core.Metric，定义你的MyMetric类。
2. 在MyMetric的构造函数中，自定义Metric组网，声明self._global_metric_state_vars私有变量。
3. 定义_calculate(global_metrics)，全局指标计算。该函数的输入globla_metrics，存储了self._global_metric_state_vars中所有中间状态变量的全局统计值。最终结果以str格式返回。

自定义Metric模版如下，你可以参考注释，或paddlerec.core.metrics下已经实现的precision_recall， auc, pairwise_pn， recall_k等指标的计算方式，自定义自己的Metric类。
```
from paddlerec.core.Metric import Metric

class MyMetric(Metric):
    def __init__(self):
        # 1. 自定义Metric组网
        ** 1. your code **

        # 2. 设置中间状态字典
        self._global_metric_state_vars = dict()
        ** 2. your code **

    def get_result(self):
        # 3. 定义训练过程中需要打印的变量，以字典格式返回
        self. _metrics = dict()
        ** 3. your code **

    def _calculate(self, global_metrics):
        # 4. 全局指标计算，global_metrics为字典类型，存储了self._global_metric_state_vars中所有中间状态变量的全局统计值。返回格式为str。
        ** your code **
```
