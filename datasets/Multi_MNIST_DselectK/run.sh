wget -c https://paddlerec.bj.bcebos.com/datasets/Multi_Mnist_Dselet_K/multi_mnist.zip
unzip multi_mnist.zip

mkdir train test
mv train.pickle ./train/train.pickle
mv test.pickle ./test/test.pickle
rm -rf multi_mnist.zip train.pickle test.pickle
