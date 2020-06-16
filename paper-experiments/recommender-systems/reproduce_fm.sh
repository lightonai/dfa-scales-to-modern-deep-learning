#!/usr/bin/env bash
GPU_ID=0
# BP - FM is only trained with BP as a baseline
python recsys.py --model fm --training-method bp \
         --learning-rate 0.0001 --gpu-id ${GPU_ID}  # AUC: 0.79148 LOGLOSS: 0.46870
