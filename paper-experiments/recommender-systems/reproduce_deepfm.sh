#!/usr/bin/env bash
GPU_ID=4
# BP
python recsys.py --model deepfm --training-method bp --learning-rate 0.0003 --dropout 0.5 \
                 --patience 5 --epoch 30 --gpu-id ${GPU_ID}  # AUC: 0.79536, LOGLOSS 0.46104
# DFA
python recsys.py --model deepfm --training-method dfa --learning-rate 0.00005 --dropout 0.5 \
                 --patience 5 --epoch 30 --gpu-id ${GPU_ID}  # AUC: 0.79558, LOGLOSS 0.46238
# SHALLOW
python recsys.py --model deepfm --training-method shallow --learning-rate 0.0001 --dropout 0.1 \
                 --patience 5 --epoch 30 --gpu-id ${GPU_ID}  # AUC: 0.79202, LOGLOSS 0.46824
