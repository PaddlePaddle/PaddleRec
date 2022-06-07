mkdir whole_data
mkdir whole_data/train && mkdir whole_data/test
wget https://paddlerec.bj.bcebos.com/datasets/IPREC/3_days.zip
unzip 3_days.zip
mv train.jsonl whole_data/train
mv test.jsonl whole_data/test
