"""
Utils for training a language model.
Code may be from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
 or https://graviraja.github.io/transformerimp/
 or http://nlp.seas.harvard.edu/2018/04/03/attention.html + OpenNMT (same folks)
 or myself.
Author: François Boniface
"""

import os
import torch
import torchtext
from tqdm import tqdm
import youtokentome as yttm


DATASET_DIR = {
        'wikitext2': torchtext.datasets.WikiText2,
        'wikitext103': torchtext.datasets.WikiText103
    }


def get_datasets(dataset):
    return DATASET_DIR[dataset]


def count_same_name(dir, name):
    count = 0
    for subfolder in os.listdir(dir):
        subfolder_name = '_'.join(subfolder.split('_')[:-1])
        if subfolder_name == name:
            count += 1
    return count


def bpe_tokenize(text, bpe_model):
    """Implements byte-pair-encoding using a saved BPE model."""
    return bpe_model.encode([text], output_type=yttm.OutputType.SUBWORD)[0]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def batchify(data, field, batch_size, device):
    data = field.numericalize([data.examples[0].text])
    n_batch = data.size(0) // batch_size
    data = data.narrow(0, 0, n_batch * batch_size)  # Trim off any extra elements that wouldn't cleanly fit
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i, chunk_length):
    seq_len = min(chunk_length, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def run_epoch(model, train_data, criterion, optimizer, vocab_size, chunk_length, alignment=None):
    model.train()
    total_loss = 0
    for i in tqdm(range(0, train_data.size(0) - 1, chunk_length)):
        data, targets = get_batch(train_data, i, chunk_length)
        if i == 0 and alignment:
            print("Measuring alignment...")
            loss_function = lambda output, target: criterion(output.view(-1, vocab_size), target)
            _, epoch_alignment = alignment.measure_alignment(data, targets, loss_function)
            print(f"{i} -- alignment (mean, std): {epoch_alignment}.")
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, vocab_size), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += len(data) * loss.item()

    if alignment:
        return total_loss / (len(train_data) - 1), epoch_alignment
    else:
        return total_loss / (len(train_data) - 1)


def evaluate(model, test_data, criterion, vocab_size, chunk_length):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(0, test_data.size(0) - 1, chunk_length):
            data, targets = get_batch(test_data, i, chunk_length)
            output = model(data)
            output_flat = output.view(-1, vocab_size)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(test_data) - 1)


class NoamOpt:
    """Wrapper to optimizer that implements lr scheduling"""
    def __init__(self, d_model, factor, warmup, optimizer):
        self.optimizer = optimizer
        self.warmup = warmup
        self.factor = factor
        self.d_model = d_model
        self._step = 0
        self._rate = 0

    @property
    def n_steps(self):
        return self._step

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        """Updates parameters and lr"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implements lr schedule from Transformer paper"""
        if step is None:
            step = self._step
        term1 = step**(-0.5)
        term2 = step * self.warmup**(-1.5)
        lrate = self.d_model**(-0.5) * min(term1, term2)
        return self.factor * lrate

    def state_dict(self):
        return {
            'factor': self.factor,
            'warmup': self.warmup,
            'step': self._step,
            'd_model': self.d_model,
            'rate': self._rate,
            'optimizer_state_dict': self.optimizer.state_dict()
        }
