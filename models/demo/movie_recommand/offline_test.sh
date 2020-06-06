## modify config.yaml to infer mode at first

cd recall
python -m paddlerec.run -m ./config.yaml
cd ../rank
python -m paddlerec.run -m ./config.yaml
cd ..

echo "recall offline test result:"
python parse.py recall_offline recall/infer_result
echo "rank offline test result:"
python parse.py rank_offline rank/infer_result
