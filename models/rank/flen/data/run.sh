mkdir train

for i in `ls ./train_data`
do
    cat train_data/$i | python get_slot_data.py > train/$i
done
