
# test file reader

python file_reader.py < utils/part-0

# test cpp reader 

g++ -o parser parser.cpp && ./parser < utils/part-0

# test tfrecord reader

python tfrecord_reader.py < utils/wd.tfrecord

# test kafka reader

export KAFKA_HOSTS=1.1.1.1
export KAFKA_GID=xxx
export KAFKA_TOPICS=wide-and-deep-data

python kafka_reader.py
