#!/usr/bin/env bash

python export_model.py
./run_tf_serving_in_docker.sh
python cold_war_tornado.py