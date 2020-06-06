cd data
mkdir online_user/test
python process_ml_1m.py data_recall > online_user/test/data.txt

## modify recall/config.yaml to online_infer mode
cd ../recall
python -m paddlerec.run -m ./config.yaml
cd ../
python parse.py recall_online recall/infer_result
