#!/bin/bash

export FLAGS_LAUNCH_BARRIER=0
fleetrun --worker_num=1 --server_num=1 tools/static_paddlebox_trainer.py -m models/rank/dnn/config_paddlebox.yaml &
