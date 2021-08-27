#!/bin/bash

#protoc *.proto --cpp_out .

rm -rf predictor.log
rm -rf std.log
rm -rf performance.txt
rm -rf cube.result

mkdir -p bin
cd bin
rm -rf *
cd ..
mkdir -p build
cd build
rm -rf *

cmake .. && make -j 10

cd ../bin
#for threadNum in 1
#do
#    for batchSize in 1 2 4 8 16 24 32 64 128
#    do
#        echo "++++ executing task : threadNum - $threadNum, batchSize - $batchSize"
#        ./main --flagfile ../user.flags -threadNum $threadNum -batchSize $batchSize
#    done
#done

#for threadNum in 1 2 4 8 16 24 32 64
#do
#    for batchSize in 1
#    do
#        echo "++++ executing task : threadNum - $threadNum, batchSize - $batchSize"
#        ./main --flagfile ../user.flags -threadNum $threadNum -batchSize $batchSize
#    done
#done

for threadNum in 1
do
    for batchSize in 1
    do
        echo "++++ executing task : threadNum - $threadNum, batchSize - $batchSize"
        ./main --flagfile ../user.flags -threadNum $threadNum -batchSize $batchSize
    done
done
