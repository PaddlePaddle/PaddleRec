type=$1

if [[ ${type} = "demo" ]]
then
    wget --no-check-certificate https://paddlerec.bj.bcebos.com/tree-based/data/demo.dat -O data/demo.dat
    python data/data_cutter.py --input "./data/demo.dat" --train "./data/demo_train.csv" --test "./data/demo_test.csv" --number 10
    python data/data_generator.py --train_file "data/demo_train.csv" --test_file "data/demo_test.csv" --item_cate_filename "demo_data/ItemCate.txt" --stat_file "demo_data/Stat.txt" --train_dir "demo_data/train_data" --test_dir "demo_data/test_data" --sample_dir "demo_data/samples" --parall 16 --train_sample_seg_cnt 20 --seq_len 70 --min_seq_len 8
    python builder/tree_index_builder.py --mode "by_category" --branch 2 --input "demo_data/ItemCate.txt" --output "demo_data/tree.pb"
elif [[ ${type} = "user_behaviour" ]]
then
    echo "ub"
fi
