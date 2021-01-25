wget https://paddle-tagspace.bj.bcebos.com/data.tar
tar -xvf data.tar
mkdir raw_big_train_data
mkdir raw_big_test_data
mv train.csv raw_big_train_data
mv test.csv raw_big_test_data
python3 text2paddle.py raw_big_train_data/ raw_big_test_data/ train_big_data test_big_data big_vocab_text.txt big_vocab_tag.txt
