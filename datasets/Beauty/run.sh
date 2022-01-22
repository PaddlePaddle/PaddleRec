mkdir data
wget https://paddlerec.bj.bcebos.com/datasets/Beauty/beauty.txt
mv beauty.txt ./data/
python data_augment_candi_gen.py
