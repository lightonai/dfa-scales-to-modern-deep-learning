import argparse
import datetime
from pathlib import Path
import os
import os.path as path
import time

import numpy as np
import yaml  # pyyaml
import torch
import tqdm
from torch.utils.data import random_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchfm.dataset.criteo import CriteoDataset
from torchfm.dataset.movielens import MovieLens1MDataset

from torchfm.model.afi import AutomaticFeatureInteractionModel
from torchfm.model.dfm import DeepFactorizationMachineModel
from torchfm.model.fm import FactorizationMachineModel
from torchfm.model.dcn import DeepCrossNetworkModel
from torchfm.model.afn import AdaptiveFactorizationNetwork

from tinydfa.alignment import AlignmentMeasurement

from dfa_models import DFADeepFactorizationMachineModel, DFADeepCrossNetworkModel, \
    DFAAutomaticFeatureInteractionModel, DFAAdaptiveFactorizationNetwork


def get_device(gpu_id):
    if not torch.cuda.is_available():
        raise OSError("CUDA not found!")

    device = torch.device(f'cuda:{gpu_id}')

    return device


def get_loaders(dataset_dir, batch_size):
    dataset_path = path.join(dataset_dir, 'criteo', 'train.txt')
    dataset = CriteoDataset(dataset_path, cache_path=str(Path.home()) + '/.criteo')

    train_length, validation_length = int(len(dataset) * 0.8), int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - validation_length
    train_dataset, validation_dataset, test_dataset = random_split(dataset,
                                                                   (train_length, validation_length, test_length))

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    features_dimension = dataset.field_dims

    return features_dimension, (train_data_loader, validation_data_loader, test_data_loader)


def get_model(model_name, features_dimension, dropout, device):
    if model_name == 'deepfm':
        model = DFADeepFactorizationMachineModel(features_dimension, embed_dim=10, mlp_dims=(400, 400, 400),
                                                 dropout=dropout, training_method='dfa')
    elif model_name == 'deepcross':
        model = DFADeepCrossNetworkModel(features_dimension, embed_dim=16, num_layers=6, mlp_dims=(1024, 1024),
                                         dropout=dropout, training_method='dfa')
    elif model_name == 'autoint':
        model = DFAAutomaticFeatureInteractionModel(features_dimension, embed_dim=16, atten_embed_dim=32,
                                                    num_heads=2, num_layers=3, mlp_dims=(400, 400, 400),
                                                    dropouts=(dropout, dropout, dropout),
                                                    training_method='dfa')
    elif model_name == 'afn':
        model = DFAAdaptiveFactorizationNetwork(features_dimension, embed_dim=10, LNN_dim=1500,
                                                mlp_dims=(400, 400, 400), dropouts=(dropout, dropout, dropout),
                                                training_method='dfa')
    return model.to(device)


def train(model, alignment, optimizer, data_loader, criterion, device, verbose, print_every=1000):
    model.train()

    accumulated_loss = 0
    for i, (features, targets) in enumerate(tqdm.tqdm(data_loader, mininterval=1.0, disable=(not verbose))):
        features, targets = features.to(device), targets.to(device)

        if i == len(data_loader) - 1:
            loss_function = lambda output, target: criterion(output, target.float())
            _, epoch_alignment = alignment.measure_alignment(features, targets, loss_function)
            optimizer.zero_grad()  # Alignment measurement generates gradients.
            print(f"Alignment (mean, std): {epoch_alignment}.")

        optimizer.zero_grad()

        predictions = model(features)
        loss = criterion(predictions, targets.float())
        accumulated_loss += loss.item()

        loss.backward()

        # use gradient clip norm 100 if DeepCross model
        if isinstance(model, DeepCrossNetworkModel) or isinstance(model, DFADeepCrossNetworkModel):
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=100.)

        optimizer.step()

        if verbose and (i + 1) % print_every == 0:
            print(f" -- training loss: {accumulated_loss / print_every:.5f}")
            accumulated_loss = 0


def evaluate(model, data_loader, criterion, device):
    model.eval()

    accumulated_loss = 0
    targets_for_eval, predictions_for_eval = [], []
    with torch.no_grad():
        for features, targets in tqdm.tqdm(data_loader, mininterval=1.0):
            features, targets = features.to(device), targets.to(device)

            predictions = model(features)
            targets_for_eval.extend(targets.tolist())
            predictions_for_eval.extend(predictions.tolist())

            loss = criterion(predictions, targets.float())
            accumulated_loss += loss.item()

    return accumulated_loss / len(data_loader), roc_auc_score(targets_for_eval, predictions_for_eval)


def main(model_name, dataset_dir, epoch, learning_rate, batch_size, dropout,
         weight_decay, patience, gpu_id, save_dir, verbose):

    device = get_device(gpu_id)
    features_dimension, (train_data_loader, validation_data_loader, test_data_loader) = get_loaders(dataset_dir,
                                                                                                    batch_size)
    model = get_model(model_name, features_dimension, dropout, device)
    alignment = AlignmentMeasurement(model, bp_device=device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
    signature = f'{timestamp}__{model_name}'
    os.mkdir(path.join(save_dir, signature))

    writer = SummaryWriter(log_dir=path.join(save_dir, signature))

    early_stop = patience
    previous_auc, last_model_path = 0, None
    for e in range(epoch):
        train(model, alignment, optimizer, train_data_loader, criterion, device, verbose)
        loss, auc = evaluate(model, validation_data_loader, criterion, device)
        writer.add_scalar("val_loss", loss, global_step=e)
        writer.add_scalar("val_auc", auc, global_step=e)
        print(f"Epoch {e}/{epoch} -- Val. AUC: {auc:.5f}, val. loss {loss:.5f}")

        if previous_auc > auc:
            print("Early stopping checkpoint!")
            early_stop -= 1
            if early_stop == 0:
                break
        else:
            early_stop = patience
            save_path = path.join(save_dir, f'{signature}', f'V{e}_{loss:.5f}_{auc:.5f}.pt')
            torch.save(model, save_path)
            previous_auc, last_model_path = auc, save_path

    if early_stop == 0:
        model = torch.load(last_model_path).to(device)

    loss, auc = evaluate(model, test_data_loader, criterion, device)
    print(f"Test AUC: {auc:.5f}, test loss {loss:.5f}")
    writer.add_scalar("test_loss", loss)
    writer.add_scalar("test_auc", auc)
    writer.close()

    save_path = path.join(save_dir, f'{signature}', f'T_{loss:.5f}_{auc:.5f}.pt')
    torch.save(model, save_path)

    with open(path.join(save_dir, f'{signature}', "config.yaml"), "w") as stream:
        yaml.dump(config, stream)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scaling-up DFA -- RecSys Experiments')
    parser.add_argument('--model', type=str, choices=['deepfm', 'deepcross', 'autoint', 'afn'], default='deepfm',
                        help='model to use, choose from Deep Factorization Machine (dfm), '
                             'Deep & Cross Network (deepcross), '
                             'Automatic Feature Interaction (autoint), '
                             'and Adaptative Factorization Network (afn) (default: deepfm)')
    parser.add_argument('--dataset-dir', type=str, default=str(Path.home()),
                        help='path to root directory containing criteo or ml-1m datasets (default: /data)')

    parser.add_argument('--epoch', type=int, default=15,
                        help='number of epochs for which to train (default: 15)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='learning rate for ADAM optimizer (default: 1e-3)')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='training batch size (default: 2048)')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='weight decay (L2 regularization) (default: 0)')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='dropout probability (default: 0.)')
    parser.add_argument('--patience', type=int, default=3,
                        help='Number of iters ith no improvement to wait before early stopping')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='id of CUDA GPU to use (default: 0)')
    parser.add_argument('--save-dir', type=str, default='results',
                        help='path to root directory for saving results')
    parser.add_argument('--verbose', action='store_true', help='shows more messages when running the script')

    args = parser.parse_args()

    config = vars(args)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    main(args.model, args.dataset_dir, args.epoch, args.learning_rate,
         args.batch_size, args.dropout, args.weight_decay, args.patience, args.gpu_id, args.save_dir, args.verbose)
