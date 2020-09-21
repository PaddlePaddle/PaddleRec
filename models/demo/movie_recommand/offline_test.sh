## modify config.yaml to infer mode at first

echo "Recall offline test ..."
echo "Model config at models/demo/movie_recommand/recall/config_offline_test.yaml"
python -m paddlerec.run -m ./recall/config_test_offline.yaml 

echo "Rank offline test ..."
echo "Model config at models/demo/movie_recommand/rank/config_offline_test.yaml"
python -m paddlerec.run -m ./rank/config_test_offline.yaml 

echo "recall offline test result:"
python parse.py recall_offline recall/infer_result

echo "rank offline test result:"
python parse.py rank_offline rank/infer_result
