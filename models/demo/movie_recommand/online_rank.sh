cd data
python process_ml_1m.py data_rank > online_user/test/data.txt

## modify recall/config.yaml to online_infer mode
cd ../rank
python -m paddlerec.run -m ./config.yaml
cd ../
python parse.py rank_online rank/infer_result
