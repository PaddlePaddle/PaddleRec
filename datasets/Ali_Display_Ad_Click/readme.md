# Ali_Display_Ad_Click数据集
[Ali_Display_Ad_Click](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56)是阿里巴巴提供的一个淘宝展示广告点击率预估数据集

## 原始数据集介绍
- 原始样本骨架raw_sample：淘宝网站中随机抽样了114万用户8天内的广告展示/点击日志（2600万条记录），构成原始的样本骨架
1. user：脱敏过的用户ID；
2. adgroup_id：脱敏过的广告单元ID；
3. time_stamp：时间戳；
4. pid：资源位；
5. nonclk：为1代表没有点击；为0代表点击；
6. clk：为0代表没有点击；为1代表点击；

```
user,time_stamp,adgroup_id,pid,nonclk,clk
581738,1494137644,1,430548_1007,1,0
```

- 广告基本信息表ad_feature：本数据集涵盖了raw_sample中全部广告的基本信息
1. adgroup_id：脱敏过的广告ID；
2. cate_id：脱敏过的商品类目ID；
3. campaign_id：脱敏过的广告计划ID；
4. customer: 脱敏过的广告主ID；
5. brand：脱敏过的品牌ID；
6. price: 宝贝的价格
```
adgroup_id,cate_id,campaign_id,customer,brand,price
63133,6406,83237,1,95471,170.0
```

- 用户基本信息表user_profile：本数据集涵盖了raw_sample中全部用户的基本信息
1. userid：脱敏过的用户ID；
2. cms_segid：微群ID；
3. cms_group_id：cms_group_id；
4. final_gender_code：性别 1:男,2:女；
5. age_level：年龄层次； 1234
6. pvalue_level：消费档次，1:低档，2:中档，3:高档；
7. shopping_level：购物深度，1:浅层用户,2:中度用户,3:深度用户
8. occupation：是否大学生 ，1:是,0:否
9. new_user_class_level：城市层级
```
userid,cms_segid,cms_group_id,final_gender_code,age_level,pvalue_level,shopping_level,occupation,new_user_class_level 
234,0,5,2,5,,3,0,3
```

- 用户的行为日志behavior_log：本数据集涵盖了raw_sample中全部用户22天内的购物行为
1. user：脱敏过的用户ID；
2. time_stamp：时间戳；
3. btag：行为类型, 包括以下四种：(pv:浏览),(cart:加入购物车),(fav:喜欢),(buy:购买)
4. cate：脱敏过的商品类目id；
5. brand: 脱敏过的品牌id；
```
user,time_stamp,btag,cate,brand
558157,1493741625,pv,6250,91286
```

## 预处理数据集介绍
将原始数据集中四个文件汇总到一个文件中，行程可以被reader直接读取的数据集。
数据集一共267列。使用时间戳来划分原始数据集中的raw_sample.csv，用前面7天的做训练样本（20170506-20170512），用第8天的做测试样本（20170513）。[0:150]列是根据raw_sample.csv每行记录查找对应的50条历史数据，[150:200]列是[50:100]的mask，[200:250]列是[100:150]的mask，[250:259]列为user_profile.csv中的各项特征，[259:265]列为ad_feature.csv中的各项特征，最后一列[266]为用户是否点击。
详细的[数据处理](https://aistudio.baidu.com/aistudio/projectdetail/1805731)过程点此查看。
