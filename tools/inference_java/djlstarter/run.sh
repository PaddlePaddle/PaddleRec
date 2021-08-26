echo 'stress test beginning ......'
rm -rf log.txt

trainingFile="/ssd3/wangbin44/PaddleRec/tools/inference_java/djlstarter/src/main/java/data/out_test.1"
modelFile="/ssd3/wangbin44/PaddleRec/tools/inference_java/djlstarter/src/main/java/data/rec_inference.zip"

cpuRatio=1.0
iteration=1000
outPerformanceFile="performance.txt"

for threadNum in 1 16
do
echo "executing task ++++++ threadNum: $threadNum, batchSize: $batchSize"
    for batchSize in 32
    do
        ./gradlew infer --args="-t $threadNum -bsz $batchSize -cr $cpuRatio -it $iteration -op $outPerformanceFile -inputdata $trainingFile -modelFile $modelFile"
    done
done
