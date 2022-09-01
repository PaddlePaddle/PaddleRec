sh download.sh
mkdir slot_train_data_full
mkdir slot_test_data_full

python preprocess.py --source_data ./criteo.data --output_path=./Criteo
python stratifiedKfold.py
python scale.py
python convert2txt.py
