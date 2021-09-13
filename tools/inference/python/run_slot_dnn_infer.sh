
#rm -rf performance.txt
rm -rf slot_dnn_infer.log

trainingFile=../data/out_test.1
modelFile=../model/rec_inference.pdmodel
paramFile=../model/rec_inference.pdiparams
performanceFile=performance.txt
iteration_num=10
test_predictor=True

#for threadNum in 1 2 4 8 16 24 32 64
#do
#    for batchSize in 1
#    do
#        echo "++++ executing task : threadNum - $threadNum, batchSize - $batchSize"
#        python3 -u slot_dnn_infer_dataloader.py --thread_num $threadNum --batchsize $batchSize --iteration_num $iteration_num --reader_file $trainingFile --model_file $modelFile --params_file $paramFile --performance_file $performanceFile --test_predictor $test_predictor
#    done
#done

for threadNum in 1 
do
    for batchSize in 1
    do
        echo "++++ executing task : threadNum - $threadNum, batchSize - $batchSize"
        python3 -u slot_dnn_infer_dataloader.py --thread_num $threadNum --batchsize $batchSize --iteration_num $iteration_num --reader_file $trainingFile --model_file $modelFile --params_file $paramFile --performance_file $performanceFile --test_predictor $test_predictor
    done
done

#for threadNum in 64
#do
#    for batchSize in 1
#    do
#        echo "++++ executing task : threadNum - $threadNum, batchSize - $batchSize"
#        python3 -u slot_dnn_infer_dataloader.py --thread_num $threadNum --batchsize $batchSize --iteration_num $iteration_num --reader_file $trainingFile --model_file $modelFile --params_file $paramFile --performance_file $performanceFile --test_predictor $test_predictor
#    done
#done
