sh download.sh	

mkdir slot_train_data_full	
for i in `ls ./train_data_full`	
do	
    cat train_data_full/$i | python get_slot_data.py > slot_train_data_full/$i	
done	

mkdir slot_test_data_full	
for i in `ls ./test_data_full`	
do	
    cat test_data_full/$i | python get_slot_data.py > slot_test_data_full/$i	
done	

mkdir slot_train_data	
for i in `ls ./train_data`	
do	
    cat train_data/$i | python get_slot_data.py > slot_train_data/$i	
done	

mkdir slot_test_data	
for i in `ls ./test_data`	
do	
    cat test_data/$i | python get_slot_data.py > slot_test_data/$i	
done
