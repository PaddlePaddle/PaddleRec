mkdir train_data
mkdir test_data

wget https://paddlerec.bj.bcebos.com/esmm/traindata_10w.csv  
wget https://paddlerec.bj.bcebos.com/esmm/testdata_10w.csv 
mv traindata_10w.csv train_data
mv testdata_10w.csv test_data
