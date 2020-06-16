import argparse
import os.path as path
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.utils import train_test_split_edges

from tinydfa import DFALayer, DFA
from models import GraphEncoder, DFAGraphEncoder


def get_device(gpu_id):
    if not torch.cuda.is_available():
        raise OSError("CUDA not found!")

    device = torch.device(f'cuda:{gpu_id}')

    return device


def get_data(dataset_name, dataset_dir):
    full_names = {'cora': 'Cora', 'citeseer': 'CiteSeer', 'pubmed': 'PubMed'}
    dataset_name = full_names[dataset_name]
    dataset_path = path.join(dataset_dir, dataset_name)
    dataset = Planetoid(dataset_path, dataset_name, transform=T.NormalizeFeatures())
    return dataset


def get_model_and_optimizer(training_method, dataset_name, features_dimension, device):
    training_method_signature = 'BP' if training_method == 'bp' else 'ALT'

    if training_method_signature == 'BP':
        model = GAE(GraphEncoder(features_dimension, 16))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    else:
        model = GAE(DFAGraphEncoder(features_dimension, 16, training_method=training_method))
        if dataset_name == 'cora':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        elif dataset_name == 'citeseer':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        elif dataset_name == 'pubmed':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    return model.to(device), optimizer


def train(model, optimizer, x, train_pos_edge_index):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(model, x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, test_pos_edge_index, test_neg_edge_index)


def main(training_method, dataset_name, dataset_dir, gpu_id, seed):
    torch.manual_seed(seed)
    device = get_device(gpu_id)

    dataset = get_data(dataset_name, dataset_dir)
    features_dimension = dataset.num_features
    data = dataset[0].to(device)
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data)
    x, train_pos_edge_index = data.x, data.train_pos_edge_index

    model, optimizer = get_model_and_optimizer(training_method, dataset_name, features_dimension, device)

    max_epoch = 201 if dataset_name == 'citeseer' else 401
    for epoch in range(1, max_epoch):
        train(model, optimizer, x, train_pos_edge_index)
        auc, ap = test(model, x, train_pos_edge_index, data.test_pos_edge_index, data.test_neg_edge_index)
        print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scaling-up DFA -- Geometric Autoencoder')
    parser.add_argument('--training-method', type=str, choices=['bp', 'dfa', 'shallow'], default='DFA',
                        help='training method to use, choose from backpropagation (bp), '
                             'Direct Feedback Alignment (dfa), only topmost layer (shallow), (default: DFA)')

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

    main(args.training_method, args.dataset_name, args.dataset_dir, args.gpu_id, args.seed)