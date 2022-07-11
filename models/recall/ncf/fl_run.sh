ps -ef | grep python | awk '{print $2}' | xargs kill -9

#fleetrun --worker_num=10 --server_num=1 ../../../tools/static_fl_trainer.py -m config_fl.yaml
#fleetrun --worker_num=10 --server_num=1 --coordinator_num=1 ../../../tools/static_fl_trainer_with_coordinator.py -m config_fl.yaml
fleetrun --worker_num=10 --workers="127.0.0.1:9000,127.0.0.1:9001,127.0.0.1:9002,127.0.0.1:9003,127.0.0.1:9004,127.0.0.1:9005,127.0.0.1:9006,127.0.0.1:9007,127.0.0.1:9008,127.0.0.1:9009" --server_num=1 --servers="127.0.0.1:10000" --coordinator_num=1 --coordinators="127.0.0.1:10001" ../../../tools/static_fl_trainer_with_coordinator.py -m config_fl.yaml
