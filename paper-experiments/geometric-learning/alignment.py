import argparse
import os.path as path
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from sklearn.model_selection import StratifiedKFold
from torch_geometric.datasets import Planetoid

from tinydfa.alignment import AlignmentMeasurement

from models import ChebNet, DFAChebNet, GraphNet, DFAGraphNet, SplineNet, DFASplineNet, GATNet, DFAGATNet, DNANet, DFADNANet


def get_device(gpu_id):
    if not torch.cuda.is_available():
        raise OSError("CUDA not found!")

    device = torch.device(f'cuda:{gpu_id}')

    return device


def get_data(model_name, dataset_dir):
    dataset_path = path.join(dataset_dir, 'Cora')
    if model_name == 'spline':
        transform = T.TargetIndegree()
    elif model_name == 'dna':
        transform = None
    else:
        transform = T.NormalizeFeatures()
    dataset = Planetoid(dataset_path, 'Cora', transform=transform)
    return dataset


def get_model_and_optimizer(model_name, features_dimension, classes_dimensions, device):
    if model_name == 'cheb':
        model = DFAChebNet(features_dimension, classes_dimensions, 'dfa')
        alignment_model = DFAChebNet(features_dimension, classes_dimensions, 'dfa')
        optimizer = torch.optim.Adam([dict(params=model.reg_params, weight_decay=1e-4),
                                      dict(params=model.non_reg_params, weight_decay=0)], lr=0.01)
    elif model_name == 'graph':
        model = DFAGraphNet(features_dimension, classes_dimensions, 'dfa')
        alignment_model = DFAGraphNet(features_dimension, classes_dimensions, 'dfa')
        optimizer = torch.optim.Adam([dict(params=model.reg_params, weight_decay=1e-4),
                                      dict(params=model.non_reg_params, weight_decay=0)], lr=0.01)
    elif model_name == 'spline':
        model = DFASplineNet(features_dimension, classes_dimensions, 'dfa')
        alignment_model = DFASplineNet(features_dimension, classes_dimensions, 'dfa')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)
    elif model_name == 'gat':
        model = DFAGATNet(features_dimension, classes_dimensions, 1, 'dfa')
        alignment_model = DFAGATNet(features_dimension, classes_dimensions, 1, 'dfa')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    elif model_name == 'dna':
        model = DFADNANet(features_dimension, 128, classes_dimensions, num_layers=5, heads=8, groups=16,
                          training_method='dfa')
        alignment_model = DFADNANet(features_dimension, 128, classes_dimensions, num_layers=5, heads=8, groups=16,
                                    training_method='dfa')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    return model.to(device), alignment_model.to(device), optimizer


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test(model, data, model_name):
    model.eval()
    logits, accuracies = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        predictions = logits[mask].max(1)[1]
        if model_name == 'dna':
            acc = predictions.eq(data.y[mask]).sum().item() / mask.numel()
        else:
            acc = predictions.eq(data.y[mask]).sum().item() / mask.sum().item()
        accuracies.append(acc)
    return accuracies


def main(model_name, dataset_dir, gpu_id, seed):
    torch.manual_seed(seed)
    device = get_device(gpu_id)

    dataset = get_data(model_name, dataset_dir)
    features_dimension, classes_dimensions = dataset.num_features, dataset.num_classes
    data = dataset[0]
    if model_name == 'dna':
        data.train_mask = data.val_mask = data.test_mask = None
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        idx = [torch.from_numpy(i) for _, i in skf.split(data.y, data.y)]
        data.train_mask = idx[0].to(torch.long)
        data.val_mask = idx[1].to(torch.long)
        data.test_mask = torch.cat(idx[2:], dim=0).to(torch.long)
    data = data.to(device)


    model, alignment_model, optimizer = get_model_and_optimizer(model_name, features_dimension, classes_dimensions, device)
    alignment = AlignmentMeasurement(model, bp_device=device, bp_model=alignment_model)

    best_validation_accuracy = best_test_accuracy = 0
    for e in range(1, 251):
        train(model, optimizer, data)
        training_accuracy, validation_accuracy, test_accuracy = test(model, data, model_name)

        if e % 10 == 0:
            loss_function = lambda output, target: F.nll_loss(output, target)
            _, epoch_alignment = alignment.measure_alignment(data, data.y, loss_function)
            optimizer.zero_grad()  # Alignment measurement generates gradients.
            print(f"Epoch {e} -- alignment (mean, std): {epoch_alignment}.")

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_test_accuracy = test_accuracy
        print(f"Epoch: {e}, Train: {training_accuracy:.4f}, Val.: {validation_accuracy:.4f} "
              f"({best_validation_accuracy:.4f}), Test: {test_accuracy:.4f} ({best_test_accuracy:.4f})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scaling-up DFA -- Geometric Alignment Measurement')
    parser.add_argument('--model', type=str, choices=['cheb', 'graph', 'spline', 'gat', 'dna'], default='graph',
                        help='model to use, choose from Chebyshev Convolutions (cheb), Graph Convolutions (graph), '
                             'Spline Convolutions (spline), Graph Attention Networks (gat), '
                             'and Just Jump (dna) (default: graph)')

    parser.add_argument('--dataset-dir', type=str, default=str('/data'),
                        help='path to root directory containing geometric datasets (default: /data)')

    parser.add_argument('--gpu-id', type=int, default=0,
                        help='id of CUDA GPU to use (default: 0)')
    parser.add_argument('--seed', type=int, default='0',
                        help='RNG seed for reproducibility (default: 0)')

    args = parser.parse_args()

    main(args.model, args.dataset_dir, args.gpu_id, args.seed)
