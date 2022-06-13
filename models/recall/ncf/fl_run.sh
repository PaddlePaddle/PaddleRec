ps -ef | grep python | awk '{print $2}' | xargs kill -9

fleetrun --worker_num=10 --server_num=1 ../../../tools/static_fl_trainer.py -m config_fl.yaml
