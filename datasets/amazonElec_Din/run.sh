wget https://paddlerec.bj.bcebos.com/datasets/amazonelec_din/paddle_train.txt
wget https://paddlerec.bj.bcebos.com/datasets/amazonelec_din/paddle_test.txt

mkdir train
mkdir test
cp paddle_train.txt train/
cp paddle_test.txt test/
rm -f paddle_test.txt paddle_train.txt
