
wget https://paddlerec.bj.bcebos.com/datasets/lookalike/metaheac_data.zip
unzip metaheac_data.zip

mkdir train_data
mkdir test_cold_data
mkdir test_hot_data

mv train_stage1.pkl train_data
mv test_hot_stage1.pkl test_hot_data
mv test_hot_stage2.pkl test_hot_data
mv test_cold_stage1.pkl test_cold_data
mv test_cold_stage2.pkl test_cold_data

rm -rf Lookalike_data.rar
