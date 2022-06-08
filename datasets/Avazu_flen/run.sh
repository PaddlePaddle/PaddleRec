mkdir train_data_full
mkdir test_data_full
mkdir raw_file
mkdir raw_filled_file_dir
mv train ./raw_file

python preprocess.py -m data_config.yaml
