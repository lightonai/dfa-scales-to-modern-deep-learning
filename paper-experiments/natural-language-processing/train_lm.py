"""
Trains a Transformer langugage model.
Author: François Boniface
"""

import argparse
import math
import numpy as np
import os
import pickle

from radam import RAdam
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext

import time
import youtokentome as yttm
import yaml

import transformer
import utils

from tinydfa.alignment import AlignmentMeasurement


# ********** CONFIG **********
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", help="which GPU to use", type=int, default=0)

parser.add_argument("--dfa", type=str, default='none', choices=['none', 'simple', 'full'])
parser.add_argument("--no_training", help="not actually use DFA", action='store_true')
parser.add_argument("--dfa_after_vocab", help="place DFA after projection to vocab size (before by default)", action='store_true')
parser.add_argument("--dfa_embed", help="place DFA after the input embedding layer", action='store_true')
parser.add_argument("--alignment", action='store_true')

parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--patience", help=" number of consecutive epochs without improvement before early stopping", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--chunk_length", type=int, default=128)

parser.add_argument("--dmodel", help="model dimension", type=int, default=512)
parser.add_argument("--dff", help="dim_feedforward", type=int, default=2048)
parser.add_argument("--nlayers", help="number of encoder layers", type=int, default=6)
parser.add_argument("--nheads", help="number of attention heads", type=int, default=8)
parser.add_argument("--dropout", help="dropout probability", type=float, default=0.1)

parser.add_argument("--attention", type=str, default='standard', choices=['standard', 'fixed', 'dense', 'random'])
parser.add_argument("--nolayernorm", action='store_true')
parser.add_argument("--tie_embeddings", action='store_true')

parser.add_argument("--optim", type=str, default='noam', choices=['noam', 'adam', '1cycle', 'radam', 'schedule'])
parser.add_argument("--max_lr", help="max learning rate (after warmup)", type=float, default=None)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--init_lr", type=float, default=1e-7)
parser.add_argument("--warmup", help="number of warmup steps of the optimizer (increasing lr)", type=int, default=4000)
parser.add_argument("--schedule_patience", type=int, default=1)
parser.add_argument("--schedule_factor", type=int, default=0.2)

parser.add_argument("--dataset", type=str, default='wikitext103', choices=['wikitext2', 'wikitext103'])
parser.add_argument("--bpe_path", type=str, default='bpe_models/wikitext-2.bpe.32000')

parser.add_argument("--savedir", help="relative path of saving directory", type=str, default='experiments')
args = parser.parse_args()
print(args)

if args.attention == 'fixed' and args.nheads != 4:
    print("WARNING: if fixed attention heads are used, their number is fixed to 4. The nheads argument will be ignored.")
    args.nheads = 4

# ********** CREATE DIRECTORY AND SAVE CONFIG **********

dfa_suffix = '_dfa' if args.dfa != 'none' else ''
exp_name = f'LM_{args.dataset}' + dfa_suffix

exp_number = utils.count_same_name(args.savedir, exp_name) + 1
exp_dir = os.path.join(args.savedir, f'{exp_name}_{exp_number}')
print(f'Will save to {exp_dir}')
if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)
losses_save_path = os.path.join(exp_dir, 'losses.npy')

with open(os.path.join(exp_dir, 'config.yml'), 'w') as f:
    yaml.dump(args.__dict__, f)

print('Configuration file written')

# ************** CREATE DATASET, MODEL AND OPTIMIZER******************

bpe = yttm.BPE(model=args.bpe_path)
TEXT = torchtext.data.Field(tokenize=lambda x: utils.bpe_tokenize(x, bpe), lower=True)
train_txt, val_txt, test_txt = utils.get_datasets(args.dataset).splits(TEXT)
print('Dataset fetched')
TEXT.build_vocab(train_txt)
vocab_size = len(TEXT.vocab.stoi)
print(f"Unique tokens in vocabulary: {len(TEXT.vocab)}")

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

train_data = utils.batchify(train_txt, TEXT, args.batch_size, device)
val_data = utils.batchify(val_txt, TEXT, args.batch_size, device)

layernorm = not args.nolayernorm
model = transformer.LMTransformer(vocab_size, args.dmodel, args.nheads,
                                  args.dff, args.nlayers, args.dropout,
                                  tie_embeddings=args.tie_embeddings,
                                  dfa=args.dfa, no_training=args.no_training,
                                  dfa_after_vocab=args.dfa_after_vocab,
                                  dfa_embed=args.dfa_embed,
                                  attn=args.attention,
                                  layernorm=layernorm)
print(f"The model has {utils.count_parameters(model)} trainable parameters")
model.to(device)

criterion = nn.CrossEntropyLoss()
scheduler = None
betas = (args.beta1, args.beta2)
if args.optim == 'noam':
    base_optim = optim.Adam(model.parameters(), lr=args.init_lr, betas=betas, eps=1e-9)
    optimizer = utils.NoamOpt(args.dmodel, 1, args.warmup, base_optim)
elif args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr, betas=betas, eps=1e-9)
elif args.optim == '1cycle':
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr, betas=betas, eps=1e-9)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                              max_lr=args.max_lr,
                                              steps_per_epoch=len(train_data),
                                              epochs=args.max_epochs)
elif args.optim == 'radam':
    optimizer = RAdam(model.parameters(), lr=args.init_lr, betas=betas, eps=1e-9)
elif args.optim == 'schedule':
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr, betas=betas, eps=1e-9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=args.schedule_factor,
                                                     patience=args.schedule_patience)

# **************** TRAINING ******************
print('Training starts...')

alignment = None
if args.alignment:
    alignment = AlignmentMeasurement(model, torch.device(f"cuda:{args.gpu_id+1}"))
    alignments = []

train_losses, val_losses, durations = [], [], []
best_val_loss = float("inf")
epochs_wo_improvement = 0
model_save_path = None
steps = 0
for epoch in range(1, args.max_epochs + 1):
    epoch_start_time = time.time()
    if alignment:
        train_loss, align_dic = utils.run_epoch(model, train_data, criterion, optimizer, vocab_size, args.chunk_length, alignment)
        alignments.append(align_dic)
    else:
        train_loss = utils.run_epoch(model, train_data, criterion, optimizer, vocab_size, args.chunk_length)

    train_losses.append(train_loss)
    steps += len(train_data)
    val_loss = utils.evaluate(model, val_data, criterion, vocab_size, args.chunk_length)
    val_losses.append(val_loss)
    if scheduler:
        scheduler.step(val_loss)
    epoch_duration = time.time() - epoch_start_time
    durations.append(epoch_duration)
    print('-' * 89)
    print('| End of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid perplexity {:8.2f}'
          .format(epoch, epoch_duration, val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # delete previous checkpoint
        if model_save_path is not None:
            os.remove(model_save_path)
        # save current state
        model_save_path = os.path.join(exp_dir, f'model_{epoch}.pt')
        torch.save({
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'epoch': epoch,
            'step': steps
        }, model_save_path)
        epochs_wo_improvement = 0
    else:
        epochs_wo_improvement += 1

    stats = {
        'train_losses': train_losses,
        'valid_losses': val_losses,
        'mean_epoch_duration': np.mean(durations)
    }
    with open(os.path.join(exp_dir, f'stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)

    if alignment:
        with open(os.path.join(exp_dir, f'alignments.pkl'), 'wb') as f:
            pickle.dump(alignments, f)

    if epochs_wo_improvement == args.patience:
        print('Early stopping')
        break
