import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import combinations
import argparse
import math


def mask_(matrices, maskval=0.0, mask_diagonal=True):
    b, h, w = matrices.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval


class CentextAttention(nn.Module):
    def __init__(self, args, emb_dim, v_dim, out_dim, mask=False):
        super(CentextAttention, self).__init__()
        self.args = args
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.v_dim = v_dim
        self._mask = mask

        self.tokeys = nn.Linear(self.emb_dim, self.v_dim * self.out_dim, bias=False)
        self.toqueries = nn.Linear(self.emb_dim, self.v_dim * self.v_dim, bias=False)

        self.activation = nn.ReLU()

        self.softmax = nn.Softmax(dim=-1)

        self._norm_d = 1 / math.sqrt(self.v_dim)

    def forward(self, q, k, mask=None):
        b, e = k.size()
        keys = self.tokeys(k).view(b, self.v_dim, self.out_dim)
        queries = self.toqueries(q).view(b, self.v_dim, self.v_dim)

        context_weight = torch.bmm(queries, self.activation(keys))

        context_weight = context_weight * self._norm_d

        context_weight = context_weight.view(b, self.v_dim, self.out_dim)

        assert context_weight.size() == (b, self.v_dim, self.out_dim)
        if self._mask:  # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(context_weight, maskval=float('-inf'), mask_diagonal=False)

        if mask is not None:
            context_weight = context_weight.masked_fill(mask == 0, -1e9)

        context_weight = F.softmax(context_weight, dim=1)

        return context_weight


class ConceptInteractionModule(nn.Module):
    """
    Neural Network model for Agent-i's N-Qi concept interactions.
    Function takes N-Qi as input and outputs a single value f(x1, ..., xN).
    Implemented using depthwise 1d convolutions, also knows as group convs.
    """
    def __init__(
        self, args, order, num_mlps, hidden_dims=[32, 32],
            dropout=0.0, batchnorm=False, kernel='linear'
    ) -> None:
        """Initializes ConceptInteractionModule hyperparameters.
        Args:
            order: Order of N-obs concept interatctions.
                    which '1' focus on obs, '>=2' focus on agent.
            hidden_dims: Number of units in hidden layers.
            dropout: Coefficient for dropout regularization.
            batchnorm: Whether to use batchnorm or not.
            kernel: layer can either be 'exu' or 'linear'
        """
        super(ConceptInteractionModule, self).__init__()
        self.args = args
        self.order = order
        self.num_mlps = num_mlps
        self.kernel = kernel
        "Order of agent interactions has to be larger than '0', interpretable model must be 1 or 2 dim."
        assert self.order > 0
        "First layer can either be 'exu' or 'linear'."
        assert (
            self.kernel == "exu" or self.kernel == "linear"
        )

        self.layers = []
        self.hidden_dims = hidden_dims
        self._batchnorm = batchnorm
        self._dropout = dropout

        # built_networks
        self.init_networks()
        self.layers = nn.ModuleList(self.layers)

        # Last MLP layer for order agents' contributions
        self.last_mlp = nn.Conv1d(
                in_channels=self.hidden_dims[-1] * self.num_mlps,
                out_channels=1 * self.num_mlps,
                kernel_size=1,
                groups=self.num_mlps,
            )

    def init_networks(self):
        input_dim = self.order

        for dim in self.hidden_dims:
            if self.kernel == "exu":
                # First layer is ExU followed by ReLUn, U can implement by 'ExU' class
                raise TypeError("'kernel': 'exu' not implemented yet.")
            elif self.kernel == "linear":
                conv = nn.Conv1d(
                        in_channels=input_dim * self.num_mlps,
                        out_channels=dim * self.num_mlps,
                        kernel_size=1,
                        groups=self.num_mlps,
                        # bias=False
                    )
                self.layers.append(
                    conv
                )
                if self._batchnorm is True:
                    self.layers.append(nn.BatchNorm1d(dim * self.num_mlps))
                if self._dropout > 0:
                    self.layers.append(nn.Dropout(p=self.dropout))
                input_dim = dim
            else:
                raise TypeError("without corresponding kernel functions.")

    def forward(self, x):
        for i in range(len(self.hidden_dims)):
            # cov
            self.layers[i].weight.data = torch.abs(self.layers[i].weight.data)
            x = self.layers[i](x)
            if self._batchnorm is True:
                x = self.layers[i + 1](x)
            if self._dropout > 0:
                x = self.layers[i + 2](x)
            x = F.elu(x)
        self.last_mlp.weight.data = torch.abs(self.last_mlp.weight.data)
        x = self.last_mlp(x)
        return x


class QNAMer(nn.Module):
    def __init__(self, args, nary=[1, 2], kernel="linear",
                 hidden_dims=[8, 4],
                 dropout=0.0, batchnorm=False):
        super(QNAMer, self).__init__()
        self._num_subnets = 1
        self.args = args
        self.n_agents = args.n_agents
        self.kernel = kernel
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.state_dim = int(np.prod(args.state_shape))
        if hidden_dims is None:
            hidden_dims = []
        elif isinstance(hidden_dims, list):
            hidden_dims = list(hidden_dims)
        self.hidden_dims = hidden_dims
        self.unit_dim = args.unit_dim
        try:
            nary = args.mix_nary
        except:
            print("using defult mix nary:", nary)

        if nary is None:
            # if no nary specified, unary model is initialized for each obs
            self._nary_indices = {"1": list(combinations(
                    range(self.n_agents), 1)
                    )
                }
        elif isinstance(nary, list):
            # interaction for each agent starting from 2
            self._nary_indices = {
                    str(order): list(combinations(
                        range(self.n_agents), order)
                    )
                    for order in nary
                }
        elif isinstance(nary, dict):
            # select which agent interaction
            self._nary_indices = nary
        else:
            raise TypeError("'nary': None or list or dict supported")
        print("qmix _nary_indices", self._nary_indices)

        self.concept_nary_nns = nn.ModuleDict()
        for subnet in range(self._num_subnets):
            for order in self._nary_indices.keys():
                self.concept_nary_nns[self.get_name(order, subnet)] = ConceptInteractionModule(args,
                    order=int(order),
                    num_mlps=len(self._nary_indices[order]),
                    hidden_dims=self.hidden_dims,
                    dropout=self.dropout,
                    batchnorm=self.batchnorm,
                )

        self.weight_in_feature_size = (
            sum(len(self._nary_indices[order]) for order in self._nary_indices.keys())
            * self._num_subnets
        )   # sum order

        self.query_embedding = nn.Linear(self.state_dim, self.args.hypernet_embed)
        self.key_embedding = nn.Linear(self.args.rnn_hidden_dim*self.n_agents, self.args.hypernet_embed)

        self.hyper_w = CentextAttention(
            args,
            self.args.hypernet_embed,
            self.weight_in_feature_size,
            1,
        )

        self.hyper_b_final = nn.Sequential(
            nn.Linear(self.state_dim, self.args.hypernet_embed),
            nn.ReLU(),
            nn.Linear(self.args.hypernet_embed, self.args.mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(self.args.mixing_embed_dim, 1),
        )

    def get_name(self, order, subnet):
        return f"ord{order}_net{subnet}"

    def forward(self, agent_qs, states, latent, test=False):
        bs = agent_qs.size(0)
        t = agent_qs.size(1)

        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, self.n_agents, 1)
        latent = latent.view(-1, self.args.rnn_hidden_dim*self.n_agents)

        B = agent_qs.size(0)

        out_nn = []
        for subnet in range(self._num_subnets):
            for order in self._nary_indices.keys():
                input_domain = agent_qs[:, self._nary_indices[order], :]
                out_domain = self.concept_nary_nns[self.get_name(order, subnet)](
                    input_domain.view(B, -1 ,1)
                ).squeeze(-1)  # B, q
                out_nn.append(
                    out_domain.view(B, -1)  # B, q
                )
        out_nn = torch.cat(out_nn, dim=-1)
        out_nn = out_nn.view(B, 1, self.weight_in_feature_size)
        # y = torch.sum(out_nn, dim=-1)

        # layer for NAM weight
        context_query = self.query_embedding(states)
        context_key = self.key_embedding(latent)

        w = self.hyper_w(context_query, context_key).view(B, self.weight_in_feature_size, 1)
        v = self.hyper_b_final(states).view(B, 1, 1)

        # Compute final output
        y = torch.bmm(out_nn, w) + v
        # y = torch.sum(agent_qs.view(-1, self.n_agents)*w1, dim=-1).view(B, 1) + v.view(B, 1)
        q_tot = y.view(bs, t, 1)

        # non_max_filter = 1 - max_filter

        # q_tot = ((w1 * non_max_filter + max_filter) * agent_qs).sum(dim=2, keepdim=True) + V
        if test:
            return q_tot, out_nn, w, v
        else:
            return q_tot

    def calculate_entropy_loss(self, alpha):
        alpha = torch.clamp(alpha, min=1e-4)
        entropy_loss = - (alpha * torch.log2(alpha)).sum(-1).mean()/self.n_agents
        return entropy_loss * 1e-5  # self.args.entropy_loss_weight


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unit Testing')
    parser.add_argument('--n_agents', default='5', type=int)
    parser.add_argument('--state_shape', default=32, type=int)
    parser.add_argument('--unit_dim', default=4, type=int)
    parser.add_argument('--mixing_embed_dim', default=32, type=int)
    parser.add_argument('--rnn_hidden_dim', default=16, type=int)
    parser.add_argument('--hypernet_embed', default=64, type=int)
    args = parser.parse_args()

    bs = 13*args.n_agents
    s = torch.randn(bs, args.n_agents*4+12).cpu()
    z = torch.randn(bs, args.n_agents*args.rnn_hidden_dim).cpu()
    net = QNAMer(args)
    for i in range(3):
        q = torch.ones(bs, 5).cpu()
        q_tot = net.forward(q, s, z).squeeze(-1)

    inp = torch.randn(bs, 5, 3)
    inp = F.pad(inp, [1, 0, 0, 0])
    inp[:,:, [0, 1]] = inp[:, :, [1, 0]]
