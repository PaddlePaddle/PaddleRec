set -v
wget https://paddlerec.bj.bcebos.com/datasets%2FMIND%2FMINDlarge_train.zip
wget https://paddlerec.bj.bcebos.com/datasets%2FMIND%2FMINDlarge_dev.zip
wget https://paddlerec.bj.bcebos.com/datasets%2FMIND%2FMINDlarge_test.zip
unzip datasets%2FMIND%2FMINDlarge_train.zip  -d ./train_raw
unzip datasets%2FMIND%2FMINDlarge_dev.zip  -d ./dev_raw
unzip datasets%2FMIND%2FMINDlarge_test.zip  -d ./test_raw
mkdir kkk
python3 combine.py
python3 preprocess.py --build_dict --build_dict_corpus_dir kkk --dict_path test_build_dict  
python3 preprocess.py --filter_corpus --dict_path test_build_dict --input_corpus_dir kkk --output_corpus_dir convert_text8 --min_count 5 --downsample 0.001
mkdir train
mkdir test
mkdir dev
python3 make_article.py
cp article.txt train/
cp article.txt dev/
cp article.txt test/
rm -rf convert_text8 dev_raw test_raw train_raw kkk
mkdir dict
mv cate_map dict/cate_map
mv sub_cate_map dict/sub_cate_map
mv test_build_dict_word_to_id_ dict/word_dict
rm article.txt news_backup.tsv test_build_dict
python3 generate_yaml.py
