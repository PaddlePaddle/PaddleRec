cd recall
python -m paddlerec.run -m ./config.yaml
cd ../rank
python -m paddlerec.run -m ./config.yaml &> train_log &
cd ..

echo "recall offline test: "
python infer_analys
