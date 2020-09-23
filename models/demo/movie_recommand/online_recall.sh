cd data
echo "Create online test data ..."
mkdir online_user/test
python process_ml_1m.py data_recall > online_user/test/data.txt

cd ..
echo "Recall online test ..."
echo "Model config at models/demo/movie_recommand/recall/config_online_test.yaml"
python -m paddlerec.run -m ./recall/config_test_online.yaml
python parse.py recall_online recall/infer_result
