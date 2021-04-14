# How to train naml on kunlun

## Prepare kunlun environment
[Paddle installation for machines with Kunlun XPU card](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0-rc1/install/install_Kunlun_zh.html)

## Prepare data
```shell
cd PaddleRec/datasets/MIND/data
bash run.sh
```

## Train
```shell
# set kunlun card id
export FLAGS_selected_xpus=0
# enable convolution autotune
export XPU_CONV_AUTOTUNE=2

cd PaddleRec/models/rank/naml 
python3.7 -u ../../../tools/trainer.py -m config_bigdata_kunlun.yaml
```


## Eval
```shell
# set kunlun card id
export FLAGS_selected_xpus=0
# enable convolution autotune
export XPU_CONV_AUTOTUNE=2

cd PaddleRec/models/rank/naml 
python3.7 -u ../../../tools/infer.py -m config_bigdata_kunlun.yaml
```
