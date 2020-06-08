cd recall
python -m paddlerec.run -m ./config.yaml &> log &
cd ../rank
python -m paddlerec.run -m ./config.yaml &> log &
cd ..
