#!/usr/bin/env bash
GPU_ID=7
# BP
python recsys.py --model afn --training-method bp --learning-rate 0.0004 --dropout 0.3 \
                 --epoch 15 --gpu-id ${GPU_ID}  # AUC: 0.79331, test loss 0.46299
# DFA
python recsys.py --model afn --training-method dfa --learning-rate 0.0003 --dropout 0. \
                 --epoch 15 --gpu-id ${GPU_ID}  # AUC: 0.79240, test loss 0.46210
# SHALLOW
python recsys.py --model afn --training-method shallow --learning-rate 0.0004 --dropout 0. \
                 --patience 5 --epoch 30 --gpu-id ${GPU_ID}  # AUC: 0.78593, test loss 0.46853
