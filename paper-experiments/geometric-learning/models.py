import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, ChebConv, SplineConv, GATConv, DNAConv

from tinydfa import DFALayer, DFA


class ChebNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(num_features, 16, K=2)
        self.conv2 = ChebConv(16, num_classes, K=2)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class DFAChebNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, training_method='dfa'):
        super(DFAChebNet, self).__init__()
        self.conv1 = ChebConv(num_features, 16, K=2)
        self.dfa_1 = DFALayer()
        self.conv2 = ChebConv(16, num_classes, K=2)

        self.dfa = DFA(dfa_layers=[self.dfa_1], no_training=training_method != 'dfa')

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0., training=self.training)
        x = self.dfa_1(x)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(self.dfa(x), dim=1)


class GraphNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GraphNet, self).__init__()
        self.conv1 = GCNConv(num_features, 16, cached=True)
        self.conv2 = GCNConv(16, num_classes, cached=True)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

    def get_hidden_embeddings(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        return x


class DFAGraphNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, training_method='dfa'):
        super(DFAGraphNet, self).__init__()
        self.conv1 = GCNConv(num_features, 16, cached=True)
        self.dfa_1 = DFALayer()
        self.conv2 = GCNConv(16, num_classes, cached=True)

        self.dfa = DFA(dfa_layers=[self.dfa_1], no_training=training_method != 'dfa')

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.dfa_1(x)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(self.dfa(x), dim=1)

    def get_hidden_embeddings(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        return x


class SplineNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(SplineNet, self).__init__()
        self.conv1 = SplineConv(num_features, 16, dim=1, kernel_size=2)
        self.conv2 = SplineConv(16, num_classes, dim=1, kernel_size=2)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)


class DFASplineNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, training_method='dfa'):
        super(DFASplineNet, self).__init__()
        self.conv1 = SplineConv(num_features, 16, dim=1, kernel_size=2)
        self.dfa_1 = DFALayer()
        self.conv2 = SplineConv(16, num_classes, dim=1, kernel_size=2)

        self.dfa = DFA(dfa_layers=[self.dfa_1], no_training=training_method != 'dfa')

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = self.dfa_1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(self.dfa(x), dim=1)


class GATNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, second_layer_heads=1):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, num_classes, heads=second_layer_heads, concat=True, dropout=0.6)

    def forward(self, data):
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)


class DFAGATNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, second_layer_heads=8, training_method='dfa'):
        super(DFAGATNet, self).__init__()
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        self.dfa_1 = DFALayer()
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, num_classes, heads=second_layer_heads, concat=True, dropout=0.)

        self.dfa = DFA(dfa_layers=[self.dfa_1], no_training=training_method != 'dfa')

    def forward(self, data):
        x = F.dropout(data.x, p=0.1, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = self.dfa_1(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(self.dfa(x), dim=1)


class DNANet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=1, groups=1):
        super(DNANet, self).__init__()
        self.hidden_channels = hidden_channels
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(DNAConv(hidden_channels, heads, groups, dropout=0.8, cached=True))
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x_all = x.view(-1, 1, self.hidden_channels)
        for conv in self.convs:
            x = F.relu(conv(x_all, edge_index))
            x = x.view(-1, 1, self.hidden_channels)
            x_all = torch.cat([x_all, x], dim=1)
        x = x_all[:, -1]
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return torch.log_softmax(x, dim=1)


class DFADNANet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=1, groups=1,
                 training_method='dfa'):
        super(DFADNANet, self).__init__()
        self.hidden_channels = hidden_channels
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.dfa1 = DFALayer()
        self.convs = torch.nn.ModuleList()
        self.dfa_convs = []
        for i in range(num_layers):
            self.convs.append(DNAConv(hidden_channels, heads, groups, dropout=0., cached=True))
            self.dfa_convs.append(DFALayer())
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dfa = DFA([self.dfa1, *self.dfa_convs], no_training=training_method != 'dfa')

    def reset_parameters(self):
        self.lin1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.dfa1(F.relu(self.lin1(x)))
        #x = F.dropout(x, p=0.5, training=self.training)
        x_all = x.view(-1, 1, self.hidden_channels)
        for i, conv in enumerate(self.convs):
            x = self.dfa_convs[i](F.relu(conv(x_all, edge_index)))
            x = x.view(-1, 1, self.hidden_channels)
            x_all = torch.cat([x_all.detach(), x], dim=1)
        x = x_all[:, -1]
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return self.dfa(torch.log_softmax(x, dim=1))


class GraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class DFAGraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, training_method):
        super(DFAGraphEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.dfa_1 = DFALayer()
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)
        self.dfa = DFA(dfa_layers=[self.dfa_1], no_training=training_method != 'dfa')

    def forward(self, x, edge_index):
        x = self.dfa_1(F.relu(self.conv1(x, edge_index)))
        return self.dfa(self.conv2(x, edge_index))
