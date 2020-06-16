import torch

#from tinydfa import DFA, DFALayer, FeedbackPointsHandling
from tinydfa.light_dfa import DFA, DFALayer


class VeryTinyNeRFModel(torch.nn.Module):
    r"""Define a "very tiny" NeRF model comprising three fully connected layers.
    """

    def __init__(self, filter_size=128, num_encoding_functions=6, use_viewdirs=True):
        super(VeryTinyNeRFModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        if use_viewdirs is True:
            self.viewdir_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        else:
            self.viewdir_encoding_dims = 0
        # Input layer (default: 65 -> 128)
        self.layer1 = torch.nn.Linear(
            self.xyz_encoding_dims + self.viewdir_encoding_dims, filter_size
        )
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        # Layer 3 (default: 128 -> 4)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class MultiHeadNeRFModel(torch.nn.Module):
    r"""Define a "multi-head" NeRF model (radiance and RGB colors are predicted by
    separate heads).
    """

    def __init__(self, hidden_size=128, num_encoding_functions=6, use_viewdirs=True):
        super(MultiHeadNeRFModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        if use_viewdirs is True:
            self.viewdir_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        else:
            self.viewdir_encoding_dims = 0
        # Input layer (default: 39 -> 128)
        self.layer1 = torch.nn.Linear(self.xyz_encoding_dims, hidden_size)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        # Layer 3_1 (default: 128 -> 1): Predicts radiance ("sigma")
        self.layer3_1 = torch.nn.Linear(hidden_size, 1)
        # Layer 3_2 (default: 128 -> 1): Predicts a feature vector (used for color)
        self.layer3_2 = torch.nn.Linear(hidden_size, hidden_size)

        # Layer 4 (default: 39 + 128 -> 128)
        self.layer4 = torch.nn.Linear(
            self.viewdir_encoding_dims + hidden_size, hidden_size
        )
        # Layer 5 (default: 128 -> 128)
        self.layer5 = torch.nn.Linear(hidden_size, hidden_size)
        # Layer 6 (default: 128 -> 3): Predicts RGB color
        self.layer6 = torch.nn.Linear(hidden_size, 3)

        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x, view = x[..., : self.xyz_encoding_dims], x[..., self.xyz_encoding_dims :]
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        sigma = self.layer3_1(x)
        feat = self.relu(self.layer3_2(x))
        x = torch.cat((feat, view), dim=-1)
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.layer6(x)
        return torch.cat((x, sigma), dim=-1)


class ReplicateNeRFModel(torch.nn.Module):
    r"""NeRF model that follows the figure (from the supp. material of NeRF) to
    every last detail. (ofc, with some flexibility)
    """

    def __init__(
        self,
        hidden_size=256,
        num_layers=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
    ):
        super(ReplicateNeRFModel, self).__init__()
        # xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions

        self.dim_xyz = (3 if include_input_xyz else 0) + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = (3 if include_input_dir else 0) + 2 * 3 * num_encoding_fn_dir

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_alpha = torch.nn.Linear(hidden_size, 1)

        self.layer4 = torch.nn.Linear(hidden_size + self.dim_dir, hidden_size // 2)
        self.layer5 = torch.nn.Linear(hidden_size // 2, hidden_size // 2)
        self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        xyz, direction = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        x_ = self.relu(self.layer1(xyz))
        x_ = self.relu(self.layer2(x_))
        feat = self.layer3(x_)
        alpha = self.fc_alpha(x_)
        y_ = self.relu(self.layer4(torch.cat((feat, direction), dim=-1)))
        y_ = self.relu(self.layer5(y_))
        rgb = self.fc_rgb(y_)
        return torch.cat((rgb, alpha), dim=-1)


class PaperNeRFModel(torch.nn.Module):
    r"""Implements the NeRF model as described in Fig. 7 (appendix) of the
    arXiv submission (v0). """

    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
    ):
        super(PaperNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir

        self.layers_xyz = torch.nn.ModuleList()
        self.use_viewdirs = use_viewdirs
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz, 256))
        for i in range(1, 8):
            if i == 4:
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + 256, 256))
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))
        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir, 128))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        for i in range(8):
            if i == 4:
                x = self.layers_xyz[i](torch.cat((xyz, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        if self.use_viewdirs:
            x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        else:
            x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, alpha), dim=-1)


class FlexibleNeRFModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=4,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
    ):
        super(FlexibleNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu

    def forward(self, x):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        else:
            xyz = x[..., : self.dim_xyz]
        x = self.layer1(xyz)  # Missing a ReLU (?)
        for i in range(len(self.layers_xyz)):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != len(self.linear_layers) - 1
            ):
                x = torch.cat((x, xyz), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)


class DFAFlexibleNeRFModel(torch.nn.Module):
    def __init__(self, num_layers=4, hidden_size=128, skip_connect_every=4, num_encoding_fn_xyz=6,
                 num_encoding_fn_dir=4, include_input_xyz=True, include_input_dir=True, use_viewdirs=True,):
        super(DFAFlexibleNeRFModel, self).__init__()

        # Determine the inputs:
        include_input_xyz = 3 if include_input_xyz else 0  # Add raw xyz coordinates
        include_input_dir = 3 if include_input_dir else 0  # Add raw viewing angle (specularity)
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz  # Total xyz input: raw? + embedding

        self.use_viewdirs = use_viewdirs  # Are we using view direction? (specularity)
        if not self.use_viewdirs:
            self.dim_dir = 0
        else:
            self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir

        # Network layers
        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)  # Input layer
        self.dfa1 = DFALayer(name='dfa1')
        # First stack of layers, using only xyz coordinates:
        self.layers_xyz = torch.nn.ModuleList()
        self.dfa_xyz = torch.nn.ModuleList()
        self.skip_connect_every = skip_connect_every
        for i in range(num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                # Handle skip-connection.
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size))
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))
            self.dfa_xyz.append(DFALayer(name=f'dfa_xyz{i}'))


        if self.use_viewdirs:
            self.fc_alpha = torch.nn.Linear(hidden_size, 1)  # Transparency output at top of xyz stack

            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)  # Link between angle stack and xyz stack
            self.dfa_feat = DFALayer(name='dfa_feat')

            # Second stack of layers, using viewing angle:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )
            self.dfa_dir = DFALayer(name='dfa_dir')

            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)  # RGB color output, at top of viewing angle stack
        else:
            # If not using viewing angle, go straight to (transparency, r, g, b) output:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu

        self.dfa_layers = [self.dfa1, *self.dfa_xyz, self.dfa_feat, self.dfa_dir]
        self.dfa = DFA(self.dfa_layers) #feedback_points_handling=FeedbackPointsHandling.MINIBATCH)

    def forward(self, x):
        # Separate the xyz and viewing angle embeddings
        if self.use_viewdirs:
            xyz, view = x[..., :self.dim_xyz], x[..., self.dim_xyz:]
        else:
            xyz = x[..., :self.dim_xyz]

        x = self.dfa1(self.relu(self.layer1(xyz)))  # Go through first layer
        # Go through xyz stack:
        for i in range(len(self.layers_xyz)):
            if (i % self.skip_connect_every == 0 and i > 0 and i != len(self.linear_layers) - 1):
                # Handle skip connection
                x = torch.cat((x, xyz), dim=-1)
            x = self.dfa_xyz[i](self.relu(self.layers_xyz[i](x))) # Go through layer

        if self.use_viewdirs:
            alpha = self.fc_alpha(x)  # Output alpha (transparency value)
            # Prepare for viewing angle stack:
            feat = self.dfa_feat(self.relu(self.fc_feat(x))) # Link between xyz/viewing angle stack
            x = torch.cat((feat, view), dim=-1)  # Add viewing angle information
            for l in self.layers_dir:
                # Go through viewing angle stack (proper):
                x = self.dfa_dir(self.relu(l(x)))
            rgb = self.fc_rgb(x)  # Output rgb value
            return self.dfa(torch.cat((rgb, alpha), dim=-1))
        else:
            return self.dfa(self.fc_out(x))
