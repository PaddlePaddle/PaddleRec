mkdir big_train
mkdir big_test
wget https://paddlerec.bj.bcebos.com/datasets/dmr/dataset_full.zip
unzip dataset_full.zip
mv work/train_sorted.csv big_train/
mv work/test.csv big_test/
rm -rf work
