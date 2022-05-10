mkdir data
mkdir data/whole_data && mkdir data/whole_data/train && mkdir data/whole_data/test
train_source_path="./data/sample_train.tar.gz"
train_target_path="train_data"
test_source_path="./data/sample_test.tar.gz"
test_target_path="test_data"
cd data
echo "downloading sample_train.tar.gz......"
curl -# 'http://jupter-oss.oss-cn-hangzhou.aliyuncs.com/file/opensearch/documents/408/sample_train.tar.gz?Expires=1586435769&OSSAccessKeyId=LTAIGx40tjZWxj6q&Signature=ahUDqhvKT1cGjC4%2FIER2EWtq7o4%3D&response-content-disposition=attachment%3B%20' -H 'Proxy-Connection: keep-alive' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' -H 'Accept-Language: zh-CN,zh;q=0.9' --compressed --insecure -o sample_train.tar.gz
cd ..
echo "unzipping sample_train.tar.gz......"
tar -xzvf  ${train_source_path} -C data && rm -rf ${train_source_path}
cd data
echo "downloading sample_test.tar.gz......"
curl -# 'http://jupter-oss.oss-cn-hangzhou.aliyuncs.com/file/opensearch/documents/408/sample_test.tar.gz?Expires=1586435821&OSSAccessKeyId=LTAIGx40tjZWxj6q&Signature=OwLMPjt1agByQtRVi8pazsAliNk%3D&response-content-disposition=attachment%3B%20' -H 'Proxy-Connection: keep-alive' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' -H 'Accept-Language: zh-CN,zh;q=0.9' --compressed --insecure -o sample_test.tar.gz
cd ..
echo "unzipping sample_test.tar.gz......"
tar -xzvf  ${test_source_path} -C data && rm -rf ${test_source_path}
echo "preprocessing data......"
python process_public_data.py
mv data/ctr_cvr.train data/whole_data/train
mv data/ctr_cvr.test data/whole_data/test
