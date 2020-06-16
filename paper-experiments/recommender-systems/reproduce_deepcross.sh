#!/usr/bin/env bash
GPU_ID=0
# BP
python recsys.py --model deepcross --training-method bp --learning-rate 0.0002 --dropout 0. \
                 --batch-size 512 --gpu-id ${GPU_ID}  # AUC: 0.81036 LOGLOSS: 0.44141
# DFA
python recsys.py --model deepcross --training-method dfa --learning-rate 0.00005 --dropout 0. \
                 --batch-size 512 --gpu-id ${GPU_ID}  # AUC: 0.80090, test loss 0.45017
# SHALLOW
python recsys.py --model deepcross --training-method shallow --learning-rate 0.001 --dropout 0. \
                 --patience 5 --epoch 30 --gpu-id ${GPU_ID}  # AUC:  0.73240, test loss 0.50102
