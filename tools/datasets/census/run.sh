train_dir="train_all"
test_dir="test_all"
rm -rf $train_dir
rm -rf $test_dir
mkdir $train_dir
mkdir $test_dir
cd $test_dir
wget https://paddlerec.bj.bcebos.com/mmoe/test_data.csv
cd ../$train_dir 
wget https://paddlerec.bj.bcebos.com/mmoe/train_data.csv
echo "good"
