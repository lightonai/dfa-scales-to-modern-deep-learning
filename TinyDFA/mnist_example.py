import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from tinydfa import DFA, DFALayer, FeedbackPointsHandling

# Fully connected neural network
class MNISTFullyConnected(nn.Module):
    def __init__(self, hidden_size, training_method='DFA'):
        super(MNISTFullyConnected, self).__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)

        self.training_method = training_method
        if self.training_method in ['DFA', 'SHALLOW']:
            self.dfa1, self.dfa2 = DFALayer(), DFALayer()
            self.dfa = DFA([self.dfa1, self.dfa2], feedback_points_handling=FeedbackPointsHandling.LAST,
                           no_training=(self.training_method == 'SHALLOW'))

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        if self.training_method in ['DFA', 'SHALLOW']:
            x = self.dfa1(torch.relu(self.fc1(x)))
            x = self.dfa2(torch.relu(self.fc2(x)))
            x = self.dfa(self.fc3(x))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        return x


def train(args, train_loader, model, optimizer, device, epoch):
    model.train()
    for b, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        print(f"Training loss at batch {b}: {loss.item():.4f}", end='\r')


def test(args, test_loader, model, device, epoch):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for b, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f"Epoch {epoch}: test loss {test_loss:.4f}, accuracy {correct / len(test_loader.dataset) * 100:.2f}.")



def main(args):
    use_gpu = not args.no_gpu and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu_id}" if use_gpu else "cpu")
    torch.manual_seed(args.seed)

    gpu_args = {'num_workers': 12, 'pin_memory': True} if use_gpu else {}
    mnist_transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(args.dataset_path, train=True, download=True,
                                                                          transform=mnist_transform),
                                               batch_size=args.batch_size, shuffle=True, **gpu_args)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(args.dataset_path, train=False,
                                                                         transform=mnist_transform),
                                              batch_size=args.test_batch_size, shuffle=True, **gpu_args)

    model = MNISTFullyConnected(args.hidden_size, training_method=args.training_method).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, train_loader, model, optimizer, device, epoch)
        test(args, test_loader, model, device, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tiny-DFA MNIST Example')
    parser.add_argument('-t', '--training-method', type=str, choices=['BP', 'DFA', 'SHALLOW'], default='DFA',
                        metavar='T', help='training method to use, choose from backpropagation (BP), direct feedback '
                                          'alignment (DFA), or only topmost layer (SHALLOW) (default: DFA)')

    parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='B',
                        help='training batch size (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='B',
                        help='testing batch size (default: 1000)')

    parser.add_argument('--hidden-size', type=int, default=256, metavar='H',
                        help='hidden layer size (default: 256)')

    parser.add_argument('-e', '--epochs', type=int, default=15, metavar='E',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01, metavar='LR',
                        help='SGD learning rate (default: 0.01)')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='disables GPU training')
    parser.add_argument('-g', '--gpu-id', type=int, default=1, metavar='i',
                        help='id of the gpu to use (default: 0)')
    parser.add_argument('-s', '--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('-p', '--dataset-path', type=str, default='/data', metavar='P',
                        help='path to dataset (default: /data)')
    args = parser.parse_args()

    main(args)
