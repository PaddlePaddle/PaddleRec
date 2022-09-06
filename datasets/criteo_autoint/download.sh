wget --no-check-certificate https://fleet.bj.bcebos.com/ctr_data.tar.gz

tar -zxvf ctr_data.tar.gz

mkdir ./tmp
mv raw_data tmp
mv test_data tmp

find ./tmp -type f -name 'part*' -exec cat {} \; > criteo.data
rm -rf ./tmp
echo "Complete data download."
