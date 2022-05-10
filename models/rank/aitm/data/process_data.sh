tar zxvf sample_train.tar.gz
tar zxvf sample_test.tar.gz
python process_public_data.py
mkdir whole_data && mkdir whole_data/train && mkdir whole_data/test
mv ctr_cvr.train whole_data/train
mv ctr_cvr.test whole_data/test
