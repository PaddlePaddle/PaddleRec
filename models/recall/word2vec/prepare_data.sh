#! /bin/bash

# download train_data
mkdir raw_data
wget --no-check-certificate https://paddlerec.bj.bcebos.com/word2vec/1-billion-word-language-modeling-benchmark-r13output.tar
tar xvf 1-billion-word-language-modeling-benchmark-r13output.tar
mv 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/ raw_data/

# preprocess data
python preprocess.py --build_dict --build_dict_corpus_dir raw_data/training-monolingual.tokenized.shuffled --dict_path raw_data/test_build_dict
python preprocess.py --filter_corpus --dict_path raw_data/test_build_dict --input_corpus_dir raw_data/training-monolingual.tokenized.shuffled --output_corpus_dir raw_data/convert_text8 --min_count 5 --downsample 0.001
mkdir thirdparty
mv raw_data/test_build_dict thirdparty/
mv raw_data/test_build_dict_word_to_id_ thirdparty/

python preprocess.py --data_resplit --input_corpus_dir=raw_data/convert_text8 --output_corpus_dir=train_data

# download test data
wget --no-check-certificate https://paddlerec.bj.bcebos.com/word2vec/test_dir.tar
tar xzvf test_dir.tar -C raw_data
mv raw_data/data/test_dir test_data/
rm -rf raw_data



