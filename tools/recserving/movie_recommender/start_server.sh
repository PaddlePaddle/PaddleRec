cd proto && python run_codegen.py && cd ..
export PYTHONPATH=$PYTHONPATH:$PWD/proto
python3 um.py &
python3 cm.py &
python3 recall.py &
python3 rank.py &
python3 as.py &
