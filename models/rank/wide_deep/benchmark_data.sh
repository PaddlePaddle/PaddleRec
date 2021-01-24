echo "Begin DownLoad Criteo Data"
wget --no-check-certificate https://paddlerec.bj.bcebos.com/benchmark/criteo_benchmark_data.tar.gz
echo "Begin Unzip Criteo Data"
tar -xf criteo_benchmark_data.tar.gz
echo "Get Criteo Data Success"
