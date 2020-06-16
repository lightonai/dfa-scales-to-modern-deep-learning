import torch
import torch.nn.functional as F

from tinydfa import DFA, DFALayer, FeedbackPointsHandling

from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear
from torchfm.model.afn import LNN


class DFAMultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True, dfa_output=True, training_method='dfa'):
        super().__init__()
        self.dfa_output = dfa_output

        layers = list()
        self.dfa_layers = []
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            dfa_layer = DFALayer()
            layers.append(dfa_layer)
            self.dfa_layers.append(dfa_layer)
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
            if dfa_output:
                layers.append(DFA(dfa_layers=self.dfa_layers, feedback_points_handling=FeedbackPointsHandling.LAST,
                                  no_training=training_method != 'dfa'))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        return self.mlp(x)


class DFACrossNetwork(torch.nn.Module):
    def __init__(self, input_dim, num_layers, dfa_output=True, training_method='dfa'):
        super().__init__()
        self.dfa_output = dfa_output

        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])
        self.dfa_layers = [DFALayer() for _ in range(num_layers - 1)]
        if self.dfa_output:
            self.dfa = DFA(dfa_layers=self.dfa_layers, feedback_points_handling=FeedbackPointsHandling.Last,
                           no_training=training_method != 'dfa')

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
            if i != self.num_layers - 1:
                x = self.dfa_layers[i](x)
        return self.dfa(x) if self.dfa_output else x


class DFADeepFactorizationMachineModel(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, training_method='dfa'):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)  # Trained through FM. OK: no weights in FM.
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = DFAMultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, training_method=training_method)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.detach().view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))


class DFADeepCrossNetworkModel(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout, training_method='dfa'):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = DFACrossNetwork(self.embed_output_dim, num_layers, dfa_output=False)
        self.mlp = DFAMultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False, dfa_output=False)
        self.linear = torch.nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)

        self.dfa_embeddings = DFALayer()
        self.dfa_stack = DFALayer()
        self.dfa = DFA(dfa_layers=[*self.cn.dfa_layers, *self.mlp.dfa_layers, self.dfa_stack, self.dfa_embeddings],
                       feedback_points_handling=FeedbackPointsHandling.LAST, no_training=training_method != 'dfa')

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        embed_x = self.dfa_embeddings(embed_x)
        x_l1 = self.cn(embed_x)
        h_l2 = self.mlp(embed_x)
        x_stack = torch.cat([x_l1, h_l2], dim=1)
        x_stack = self.dfa_stack(x_stack)
        p = self.linear(x_stack)
        p = self.dfa(p)
        return torch.sigmoid(p.squeeze(1))


class DFAAutomaticFeatureInteractionModel(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, atten_embed_dim, num_heads, num_layers, mlp_dims, dropouts,
                 has_residual=True, training_method='dfa'):
        super().__init__()
        self.num_fields = len(field_dims)
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.dfa_embedding = DFALayer()
        self.atten_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)
        self.dfa_atten_embedding = DFALayer()
        self.embed_output_dim = len(field_dims) * embed_dim
        self.atten_output_dim = len(field_dims) * atten_embed_dim
        self.has_residual = has_residual
        self.mlp = DFAMultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropouts[1], dfa_output=False)
        self.self_attns = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(atten_embed_dim, num_heads, dropout=dropouts[0]) for _ in range(num_layers)
        ])
        self.dfa_self_attns = [DFALayer() for _ in range(num_layers - 1)]

        self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)

        if self.has_residual:
            self.V_res_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)

        self.dfa_cross = DFALayer()
        self.dfa = DFA(dfa_layers=[self.dfa_atten_embedding, *self.dfa_self_attns, self.dfa_cross, *self.mlp.dfa_layers,
                                   self.dfa_embedding],
                       feedback_points_handling=FeedbackPointsHandling.LAST, no_training=training_method != 'dfa')

    def forward(self, x):
        embed_x = self.embedding(x)
        embed_x = self.dfa_embedding(embed_x)
        atten_x = self.atten_embedding(embed_x)
        atten_x = self.dfa_atten_embedding(atten_x)

        cross_term = atten_x.transpose(0, 1)
        for i, self_attn in enumerate(self.self_attns):
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
            if i != len(self.self_attns) - 1:
                cross_term = cross_term.transpose(0, 1)  # head, batch, data --> batch, head, data
                cross_term = self.dfa_self_attns[i](cross_term)
                cross_term = cross_term.transpose(0, 1)  # back to attn conf.
        cross_term = cross_term.transpose(0, 1)

        if self.has_residual:
            V_res = self.V_res_embedding(embed_x)
            cross_term += V_res

        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)
        cross_term = self.dfa_cross(cross_term)

        x = self.linear(x) + self.attn_fc(cross_term) + self.mlp(embed_x.view(-1, self.embed_output_dim))

        x = self.dfa(x)

        return torch.sigmoid(x.squeeze(1))


class DFAAdaptiveFactorizationNetwork(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, LNN_dim, mlp_dims, dropouts, training_method='dfa'):
        super().__init__()
        self.num_fields = len(field_dims)
        self.linear = FeaturesLinear(field_dims)    # Linear
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)   # Embedding
        self.dfa_embed = DFALayer()
        self.LNN_dim = LNN_dim
        self.LNN_output_dim = self.LNN_dim * embed_dim
        self.LNN = LNN(self.num_fields, embed_dim, LNN_dim)
        self.dfa_lnn = DFALayer()
        self.mlp = DFAMultiLayerPerceptron(self.LNN_output_dim, mlp_dims, dropouts[0], dfa_output=False)

        self.dfa = DFA(dfa_layers=[self.dfa_lnn, *self.mlp.dfa_layers],
                       feedback_points_handling=FeedbackPointsHandling.LAST, no_training=training_method != 'dfa')

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        embed_x = self.dfa_embed(embed_x)
        lnn_out = self.LNN(embed_x)
        lnn_out = self.dfa_lnn(lnn_out)
        x = self.linear(x) + self.mlp(lnn_out)
        x = self.dfa(x)
        return torch.sigmoid(x.squeeze(1))
