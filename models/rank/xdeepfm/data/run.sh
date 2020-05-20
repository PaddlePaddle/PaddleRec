python download.py

mkdir -p slot_train_data/tr
for i in `ls ./train_data/tr`
do
    cat train_data/tr/$i | python get_slot_data.py > slot_train_data/tr/$i
done

mkdir slot_test_data/ev
for i in `ls ./test_data/ev`
do
    cat test_data/ev/$i | python get_slot_data.py > slot_test_data/ev/$i
done
