import argparse
import os.path as path
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from sklearn.model_selection import StratifiedKFold
from torch_geometric.datasets import Planetoid

from models import ChebNet, DFAChebNet, GraphNet, DFAGraphNet, SplineNet, DFASplineNet, GATNet, DFAGATNet, DNANet, DFADNANet


def get_device(gpu_id):
    if not torch.cuda.is_available():
        raise OSError("CUDA not found!")

    device = torch.device(f'cuda:{gpu_id}')

    return device


def get_data(dataset_name, model_name, dataset_dir):
    full_names = {'cora': 'Cora', 'citeseer': 'CiteSeer', 'pubmed': 'PubMed'}
    dataset_name = full_names[dataset_name]
    dataset_path = path.join(dataset_dir, dataset_name)
    if model_name == 'spline':
        transform = T.TargetIndegree()
    elif model_name == 'dna':
        transform = None
    else:
        transform = T.NormalizeFeatures()
    dataset = Planetoid(dataset_path, dataset_name, transform=transform)
    return dataset


def get_model_and_optimizer(model_name, training_method, dataset_name, features_dimension, classes_dimensions, device):
    training_method_signature = 'BP' if training_method == 'bp' else 'ALT'

    if model_name == 'cheb':
        if training_method_signature == 'ALT':
            model = DFAChebNet(features_dimension, classes_dimensions, training_method)
            optimizer = torch.optim.Adam([dict(params=model.reg_params, weight_decay=1e-4),
                                          dict(params=model.non_reg_params, weight_decay=0)], lr=0.01)
        else:
            model = ChebNet(features_dimension, classes_dimensions)
            optimizer = torch.optim.Adam([dict(params=model.reg_params, weight_decay=5e-4),
                                          dict(params=model.non_reg_params, weight_decay=0)], lr=0.01)
    elif model_name == 'graph':
        if training_method_signature == 'ALT':
            model = DFAGraphNet(features_dimension, classes_dimensions, training_method)
            optimizer = torch.optim.Adam([dict(params=model.reg_params, weight_decay=1e-4),
                                          dict(params=model.non_reg_params, weight_decay=0)], lr=0.01)
        else:
            model = GraphNet(features_dimension, classes_dimensions)
            optimizer = torch.optim.Adam([dict(params=model.reg_params, weight_decay=5e-4),
                                          dict(params=model.non_reg_params, weight_decay=0)], lr=0.01)
    elif model_name == 'spline':
        if training_method_signature == 'ALT':
            model = DFASplineNet(features_dimension, classes_dimensions, training_method)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)
        else:
            model = SplineNet(features_dimension, classes_dimensions)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)
    elif model_name == 'gat':
        if training_method_signature == 'ALT':
            model = DFAGATNet(features_dimension, classes_dimensions, 1, training_method)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        else:
            model = GATNet(features_dimension, classes_dimensions, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    elif model_name == 'dna':
        if training_method_signature == 'ALT':
            model = DFADNANet(features_dimension, 128, classes_dimensions, num_layers=5, heads=8, groups=16,
                              training_method=training_method)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        else:
            model = DNANet(features_dimension, 128, classes_dimensions, num_layers=5, heads=8, groups=16)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)

    return model.to(device), optimizer


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


def main(model_name, training_method, dataset_name, dataset_dir, gpu_id, seed):
    torch.manual_seed(seed)
    device = get_device(gpu_id)

    dataset = get_data(dataset_name, model_name, dataset_dir)
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


    model, optimizer = get_model_and_optimizer(model_name, training_method, dataset_name, features_dimension,
                                               classes_dimensions, device)

    best_validation_accuracy = best_test_accuracy = 0
    for e in range(1, 251):
        train(model, optimizer, data)
        training_accuracy, validation_accuracy, test_accuracy = test(model, data, model_name)
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_test_accuracy = test_accuracy
        print(f"Epoch: {e}, Train: {training_accuracy:.4f}, Val.: {validation_accuracy:.4f} "
              f"({best_validation_accuracy:.4f}), Test: {test_accuracy:.4f} ({best_test_accuracy:.4f})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scaling-up DFA -- Geometric Experiments')
    parser.add_argument('--model', type=str, choices=['cheb', 'graph', 'spline', 'gat', 'dna'], default='graph',
                        help='model to use, choose from Chebyshev Convolutions (cheb), Graph Convolutions (graph), '
                             'Spline Convolutions (spline), Graph Attention Networks (gat), '
                             'and Just Jump (dna) (default: graph)')

    parser.add_argument('--training-method', type=str, choices=['bp', 'dfa', 'shallow', 'random'], default='DFA',
                        help='training method to use, choose from backpropagation (bp), '
                             'Direct Feedback Alignment (dfa), only topmost layer (shallow), or random (random) '
                             '(default: DFA)')

    parser.add_argument('--dataset-name', type=str, choices=['cora', 'citeseer', 'pubmed'], default='cora',
                        help='dataset on which to train, choose between the Cora dataset (cora)'
                             'the CiteSeer dataset (citeseer), and the PubMed dataset (pubmed) (default: cora)')
    parser.add_argument('--dataset-dir', type=str, default=str('/data'),
                        help='path to root directory containing criteo or ml-1m datasets (default: /data)')

    parser.add_argument('--gpu-id', type=int, default=0,
                        help='id of CUDA GPU to use (default: 0)')
    parser.add_argument('--seed', type=int, default='0',
                        help='RNG seed for reproducibility (default: 0)')

    args = parser.parse_args()

    main(args.model, args.training_method, args.dataset_name, args.dataset_dir, args.gpu_id, args.seed)
