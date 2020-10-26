python download_preprocess.py 
mv ./deepfm%2Ffeat_dict_10.pkl2 sample_data/feat_dict_10.pkl2
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
