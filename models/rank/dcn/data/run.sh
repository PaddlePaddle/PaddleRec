python download.py
python preprocess.py

mkdir slot_train
for i in `ls ./train`
do
    cat train/$i | python get_slot_data.py > slot_train/$i
done

mkdir slot_test_valid
for i in `ls ./test_valid`
do
    cat test_valid/$i | python get_slot_data.py > slot_test_valid/$i
done
