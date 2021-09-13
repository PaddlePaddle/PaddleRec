echo 'stress test beginning ......'
rm -rf log.txt

trainingFile="../data/out_test.1"
modelFile="../model/rec_inference.zip"

cpuRatio=1.0
iteration=1000
outPerformanceFile="performance.txt"

for threadNum in 1
do
echo "executing task ++++++ threadNum: $threadNum, batchSize: $batchSize"
    for batchSize in 1
    do
        ./gradlew infer --args="-t $threadNum -bsz $batchSize -cr $cpuRatio -it $iteration -op $outPerformanceFile -inputdata $trainingFile -modelFile $modelFile"
    done
done
