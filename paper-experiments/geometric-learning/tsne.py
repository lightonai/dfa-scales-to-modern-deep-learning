import argparse
import os.path as path
import seaborn as sb
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid
from tsnecuda import TSNE

from models import GraphNet, DFAGraphNet


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


def get_model_and_optimizer(training_method, features_dimension, classes_dimensions, device):
    training_method_signature = 'BP' if training_method == 'bp' else 'ALT'

    if training_method_signature == 'ALT':
        model = DFAGraphNet(features_dimension, classes_dimensions, training_method)
        optimizer = torch.optim.Adam([dict(params=model.reg_params, weight_decay=1e-4),
                                      dict(params=model.non_reg_params, weight_decay=0)], lr=0.01)
    else:
        model = GraphNet(features_dimension, classes_dimensions)
        optimizer = torch.optim.Adam([dict(params=model.reg_params, weight_decay=5e-4),
                                      dict(params=model.non_reg_params, weight_decay=0)], lr=0.01)

    return model.to(device), optimizer


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test(model, data):
    model.eval()
    logits, accuracies = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        predictions = logits[mask].max(1)[1]
        acc = predictions.eq(data.y[mask]).sum().item() / mask.sum().item()
        accuracies.append(acc)
    return accuracies


def main(training_method, dataset_name, dataset_dir, gpu_id, seed):
    torch.manual_seed(seed)
    device = get_device(gpu_id)

    dataset = get_data(dataset_name, dataset_dir)
    features_dimension, classes_dimensions = dataset.num_features, dataset.num_classes
    data = dataset[0]
    data = data.to(device)

    model, optimizer = get_model_and_optimizer(training_method, features_dimension, classes_dimensions, device)

    best_validation_accuracy = best_test_accuracy = 0
    for e in range(1, 251):
        train(model, optimizer, data)
        training_accuracy, validation_accuracy, test_accuracy = test(model, data)
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_test_accuracy = test_accuracy
        print(f"Epoch: {e}, Train: {training_accuracy:.4f}, Val.: {validation_accuracy:.4f} "
              f"({best_validation_accuracy:.4f}), Test: {test_accuracy:.4f} ({best_test_accuracy:.4f})")

    embeddings = model.get_hidden_embeddings(data).detach().cpu().numpy()
    tsne_embeddings = TSNE(n_components=2, perplexity=20, learning_rate=100, n_iter=5000).fit_transform(embeddings)

    sb.set_style('white', {})
    tsne_plot = sb.scatterplot(x=tsne_embeddings[:, 0], y=tsne_embeddings[:, 1], hue=data.y.cpu(), legend=False,
                               edgecolor=None, palette=sb.color_palette("colorblind", n_colors=7))
    tsne_plot.set(xticks=[], yticks=[])
    tsne_plot = tsne_plot.get_figure()
    sb.despine(fig=tsne_plot, left=True, bottom=True)
    tsne_plot.savefig('/home/julienlaunay/tsne.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scaling-up DFA -- Geometric t-SNE')

    parser.add_argument('--training-method', type=str, choices=['bp', 'dfa', 'shallow', 'random'], default='DFA',
                        help='training method to use, choose from backpropagation (bp), '
                             'Direct Feedback Alignment (dfa), only topmost layer (shallow), or random (random) '
                             '(default: DFA)')

    parser.add_argument('--dataset-name', type=str, choices=['cora', 'citeseer', 'pubmed'], default='cora',
                        help='dataset on which to train, choose between the Cora dataset (cora)'
                             'the CiteSeer dataset (citeseer), and the PubMed dataset (pubmed) (default: cora)')
    parser.add_argument('--dataset-dir', type=str, default=str('/data'),
                        help='path to root directory containing geometric datasets (default: /data)')

    parser.add_argument('--gpu-id', type=int, default=0,
                        help='id of CUDA GPU to use (default: 0)')
    parser.add_argument('--seed', type=int, default='0',
                        help='RNG seed for reproducibility (only for graph model training, '
                             't-SNE reproducibility cannot be ensured) (default: 0)')

    args = parser.parse_args()

    main(args.training_method, args.dataset_name, args.dataset_dir, args.gpu_id, args.seed)
