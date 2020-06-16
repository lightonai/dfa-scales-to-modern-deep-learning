#!/usr/bin/env bash
GPU_ID=7
# BP
python recsys.py --model autoint --training-method bp --learning-rate 0.0001 --dropout 0. \
                 --epoch 15 --gpu-id ${GPU_ID}  # AUC: 0.79367, test loss 0.46200
# DFA
python recsys.py --model autoint --training-method dfa --learning-rate 0.00005 --dropout 0. \
                 --epoch 15 --gpu-id ${GPU_ID}  # AUC: 0.79055, test loss 0.45996
# SHALLOW
python recsys.py --model autoint --training-method shallow --learning-rate 0.0004 --dropout 0.5 \
                 --patience 5 --epoch 30 --gpu-id ${GPU_ID}  # AUC: 0.78603, test loss 0.46864
