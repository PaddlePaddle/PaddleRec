wget https://paddlerec.bj.bcebos.com/aistudio/aistudio_paddlerec_rank.tar.gz --no-check-certificate
tar xf aistudio_paddlerec_rank.tar.gz
wget https://paddlerec.bj.bcebos.com/aistudio/recall.dat --no-check-certificate
wget https://paddlerec.bj.bcebos.com/aistudio/users.dat --no-check-certificate
wget https://paddlerec.bj.bcebos.com/aistudio/movies.dat --no-check-certificate
python3 to_redis.py
