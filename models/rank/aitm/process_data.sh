tar zxvf data/sample_train.tar.gz -C data
tar zxvf data/sample_test.tar.gz -C data
python process_public_data.py
mkdir data/whole_data && mkdir data/whole_data/train && mkdir data/whole_data/test
mv data/ctr_cvr.train data/whole_data/train
mv data/ctr_cvr.test data/whole_data/test
