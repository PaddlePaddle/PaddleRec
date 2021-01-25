mkdir big_train
cd big_train
wget https://paddlerec.bj.bcebos.com/match_pyramid/train.txt
cd ..
mkdir big_test
cd big_test
wget https://paddlerec.bj.bcebos.com/match_pyramid/test.txt
cd ..
wget https://paddlerec.bj.bcebos.com/match_pyramid/embedding.npy
wget --no-check-certificate 'https://paddlerec.bj.bcebos.com/match_pyramid/match_pyramid_data.tar.gz'
tar -xvf ./match_pyramid_data.tar.gz
