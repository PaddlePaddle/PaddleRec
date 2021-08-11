type=$1

if [[ ${type} = "demo" ]]
then
    wget --no-check-certificate https://paddlerec.bj.bcebos.com/tree-based/data/demo.dat -O data/demo.dat
    python data/data_cutter.py --input "./data/demo.dat" --train "./data/demo_train.csv" --test "./data/demo_test.csv" --number 10
    python data/data_generator.py --train_file "data/demo_train.csv" --test_file "data/demo_test.csv" --item_cate_filename "demo_data/ItemCate.txt" --stat_file "demo_data/Stat.txt" --train_dir "demo_data/train_data" --test_dir "demo_data/test_data" --sample_dir "demo_data/samples" --parall 16 --train_sample_seg_cnt 20 --seq_len 70 --min_seq_len 8
    python builder/tree_index_builder.py --mode "by_category" --input "demo_data/ItemCate.txt" --output "demo_data/tree.pb"
elif [[ ${type} = "user_behaviour" ]]
then
    wget --no-check-certificate https://paddlerec.bj.bcebos.com/tree-based/data/UserBehavior.csv.zip -O data/UserBehavior.csv.zip
    unzip -d data/ data/UserBehavior.csv.zip
    python data/data_cutter.py --input "./data/UserBehavior.csv" --train "./data/ub_train.csv" --test "./data/ub_test.csv" --number 10000
    python data/data_generator.py --train_file "data/ub_train.csv" --test_file "data/ub_test.csv" --item_cate_filename "ub_data_new/ItemCate.txt" --stat_file "ub_data_new/Stat.txt" --train_dir "ub_data_new/train_data" --test_dir "ub_data_new/test_data" --sample_dir "ub_data_new/samples" --parall 32 --train_sample_seg_cnt 400 --seq_len 70 --min_seq_len 6
    python builder/tree_index_builder.py --mode "by_category" --input "ub_data_new/ItemCate.txt" --output "ub_data_new/tree.pb"
    echo "ub"
fi
