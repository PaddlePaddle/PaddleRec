# NCF使用的数据集

本数据集供NCF模型复现论文使用，使用的是初步处理过后的数据，分为两个数据集：ml-1m（即MovieLens数据集）和pinterest-20（即Pinterest数据集）
每个数据集分为三个文件，后缀分别为：（.test.negative），（.test.rating），（.train.rating）

在train.rating和test.rating中的数据格式为：
user_id + \t + item_id + \t + rating(用户评分) + \t + timestamp(时间戳)
在test.negative中的数据格式为：
(userID,itemID) + \t + negativeItemID1 + \t + negativeItemID2 …(包含99个negative样本)
