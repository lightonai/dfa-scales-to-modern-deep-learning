#!/bin/bash

# this script will run all the reported experiments at once in separate tmux sessions if you have 10 GPUs
# you might want to select only a few of them
# on a V100, an epoch will take around 45 min for BP and 55 min for DFA

/usr/bin/tmux new-session -d -s bpbaseline
/usr/bin/tmux send-keys -t bpbaseline "source ~/dfaenv/bin/activate" C-m
/usr/bin/tmux send-keys -t bpbaseline "python train_lm.py --gpu_id 0 --beta2 0.98" C-m


/bin/sleep 5

/usr/bin/tmux new-session -d -s bpbaselinebeta2
/usr/bin/tmux send-keys -t bpbaselinebeta2 "source ~/dfaenv/bin/activate" C-m
/usr/bin/tmux send-keys -t bpbaselinebeta2 "python train_lm.py --gpu_id 1 --beta2 0.999" C-m


/bin/sleep 5

/usr/bin/tmux new-session -d -s dfabaselinemacro
/usr/bin/tmux send-keys -t dfabaselinemacro "source ~/dfaenv/bin/activate" C-m
/usr/bin/tmux send-keys -t dfabaselinemacro "python train_lm.py --gpu_id 2 --dfa simple --beta2 0.98" C-m


/bin/sleep 5

/usr/bin/tmux new-session -d -s dfabaselinemicro
/usr/bin/tmux send-keys -t dfabaselinemicro "source ~/dfaenv/bin/activate" C-m
/usr/bin/tmux send-keys -t dfabaselinemicro "python train_lm.py --gpu_id 3 --dfa full --beta2 0.98" C-m

/bin/sleep 5

/usr/bin/tmux new-session -d -s dfaadammacro
/usr/bin/tmux send-keys -t dfaadammacro "source ~/dfaenv/bin/activate" C-m
/usr/bin/tmux send-keys -t dfaadammacro "python train_lm.py --gpu_id 4 --dfa simple --optim adam --init_lr 5e-5 --beta2 0.98" C-m


/bin/sleep 5

/usr/bin/tmux new-session -d -s dfaadammicro
/usr/bin/tmux send-keys -t dfaadammicro "source ~/dfaenv/bin/activate" C-m
/usr/bin/tmux send-keys -t dfaadammicro "python train_lm.py --gpu_id 5 --dfa full --optim adam --init_lr 5e-5 --beta2 0.98" C-m


/bin/sleep 5

/usr/bin/tmux new-session -d -s dfabeta2macro
/usr/bin/tmux send-keys -t dfabeta2macro "source ~/dfaenv/bin/activate" C-m
/usr/bin/tmux send-keys -t dfabeta2macro "python train_lm.py --gpu_id 6 --dfa simple --optim adam --init_lr 5e-5 --beta2 0.999" C-m


/bin/sleep 5

/usr/bin/tmux new-session -d -s dfabeta2micro
/usr/bin/tmux send-keys -t dfabeta2micro "source ~/dfaenv/bin/activate" C-m
/usr/bin/tmux send-keys -t dfabeta2micro "python train_lm.py --gpu_id 7 --dfa full --optim adam --init_lr 5e-5 --beta2 0.999" C-m

/bin/sleep 5

/usr/bin/tmux new-session -d -s dfaschedulemacro
/usr/bin/tmux send-keys -t dfaschedulemacro "source ~/dfaenv/bin/activate" C-m
/usr/bin/tmux send-keys -t dfaschedulemacro "python train_lm.py --gpu_id 8 --dfa simple --optim schedule --init_lr 1e-4 --beta2 0.999" C-m

/bin/sleep 5

/usr/bin/tmux new-session -d -s dfaschedulemicro
/usr/bin/tmux send-keys -t dfaschedulemicro "source ~/dfaenv/bin/activate" C-m
/usr/bin/tmux send-keys -t dfaschedulemicro "python train_lm.py --gpu_id 9 --dfa full --optim schedule --init_lr 1e-4 --beta2 0.999" C-m