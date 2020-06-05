# PaddleRec 推荐数据集格式

当你的数据集格式为`slot:feasign`这种模式，或者可以预处理为这种格式时，可以直接使用PaddleRec内置的Reader。

> Slot : Feasign 是什么？
>
> Slot直译是槽位，在推荐工程中，是指某一个宽泛的特征类别，比如用户ID、性别、年龄就是Slot，Feasign则是具体值，比如：12345，男，20岁。
> 
> 在实践过程中，很多特征槽位不是单一属性，或无法量化并且离散稀疏的，比如某用户兴趣爱好有三个：游戏/足球/数码，且每个具体兴趣又有多个特征维度，则在兴趣爱好这个Slot兴趣槽位中，就会有多个Feasign值。
>
> PaddleRec在读取数据时，每个Slot ID对应的特征，支持稀疏，且支持变长，可以非常灵活的支持各种场景的推荐模型训练。

## 数据格式说明

假如你的原始数据格式为

```bash
<label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
```

其中```<label>```表示广告是否被点击，点击用1表示，未点击用0表示。```<integer feature>```代表数值特征（连续特征），共有13个连续特征。
并且每个特征有一个特征值。
```<categorical feature>```代表分类特征（离散特征），共有26个离散特征。相邻两个特征用```\t```分隔。

假设这13个连续特征（dense slot）的name如下：

```
D1 D2 D3 D4 D4 D6 D7 D8 D9 D10 D11 D12 D13
```

这26个离散特征（sparse slot）的name如下：
```
S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 S12 S13 S14 S15 S16 S17 S18 S19 S20 S21 S22 S23 S24 S25 S26
```

那么下面这条样本（1个label + 13个dense值 + 26个feasign）
```
1 0.1 0.4 0.2 0.3 0.5 0.8 0.3 0.2 0.1 0.5 0.6 0.3 0.9 60 16 91 50 52 52 28 69 63 33 87 69 48 59 27 12 95 36 37 41 17 3 86 19 88 60
```

可以转换成：
```
label:1 D1:0.1 D2:0.4 D3:0.2 D4:0.3 D5:0.5 D6:0.8 D7:0.3 D8:0.2 D9:0.1 D10:0.5 D11:0.6 D12:0.3 D13:0.9 S14:60 S15:16 S16:91 S17:50 S18:52 S19:52 S20:28 S21:69 S22:63 S23:33 S24:87 S25:69 S26:48 S27:59 S28:27 S29:12 S30:95 S31:36 S32:37 S33:41 S34:17 S35:3 S36:86 S37:19 S38:88 S39:60
```

注意：上面各个slot:feasign字段之间的顺序没有要求，比如```D1:0.1 D2:0.4```改成```D2:0.4 D1:0.1```也可以。


## 配置

reader中需要配置```sparse_slots```与```dense_slots```，例如

```
  workspace: xxxx

  reader:
    batch_size: 2
    train_data_path: "{workspace}/data/train_data"
    sparse_slots: "label S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 S12 S13 S14 S15 S16 S17 S18 S19 S20 S21 S22 S23 S24 S25 S26"
    dense_slots: "D1:1 D2:1 D3:1 D4:1 D4:1 D6:1 D7:1 D8:1 D9:1 D10:1 D11:1 D12:1 D13:1"

  model:
    xxxxx
```

sparse_slots表示稀疏特征的列表，以空格分开。

dense_slots表示稠密特征的列表，以空格分开。每个字段的格式是```[dense_slot_name]:[dim1,dim2,dim3...]```，其中```dim1,dim2,dim3...```表示shape


配置好了之后，这些slot对应的variable在model中可以使用如下方式调用：
```
self._sparse_data_var

self._dense_data_var
```

