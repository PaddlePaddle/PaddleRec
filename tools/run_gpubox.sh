#!/bin/bash

export FLAGS_LAUNCH_BARRIER=0
fleetrun --worker_num=1 --server_num=1 tools/static_gpubox_trainer.py -m models/rank/dnn/config_gpubox.yaml
