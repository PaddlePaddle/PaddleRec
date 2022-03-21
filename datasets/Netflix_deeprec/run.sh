wget https://paddlerec.bj.bcebos.com/datasets/Netflix-DeepRec/nf_prize_dataset.tar.gz 
tar -xvf nf_prize_dataset.tar.gz
tar -xf download/training_set.tar
python netflix_data_convert.py training_set Netflix
