cd proto && python run_codegen.py && cd ..
export PYTHONPATH=$PYTHONPATH:$PWD/proto
python um.py &
python mm.py &
python recall.py &
python rank.py &
python gr.py &
