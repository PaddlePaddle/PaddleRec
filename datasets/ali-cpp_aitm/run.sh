mkdir whole_data
mkdir whole_data/train && mkdir whole_data/test
cd whole_data/train
wget https://paddlerec.bj.bcebos.com/datasets/aitm/ctr_cvr.train
cd ../test
wget https://paddlerec.bj.bcebos.com/datasets/aitm/ctr_cvr.test
cd ../../
