mkdir Data
pip3 install scipy
wget https://paddlerec.bj.bcebos.com/ncf/Data.zip
unzip Data/Data.zip -d Data/
python3 get_train_data.py --num_neg 4  --train_data_path "Data/train_data.csv"  
python3 get_test_data.py
