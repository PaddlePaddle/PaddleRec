cd data
echo "Create online test data ..."
python process_ml_1m.py data_rank > online_user/test/data.txt

cd ..
echo "Rank online test ..."
echo "Model config at models/demo/movie_recommand/rank/config_online_test.yaml"
python -m paddlerec.run -m ./rank/config_test_online.yaml
python parse.py rank_online ./rank/infer_result
