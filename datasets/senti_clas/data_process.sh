wget https://baidu-nlp.bj.bcebos.com/sentiment_classification-dataset-1.0.0.tar.gz
tar -zxvf sentiment_classification-dataset-1.0.0.tar.gz
cp ./data/preprocess.py ./senta_data/
cd senta_data/
python preprocess.py
mkdir train
mv train.txt train
mkdir test
mv dev.txt  test
cd ..
