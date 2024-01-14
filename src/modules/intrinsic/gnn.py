import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.set_default_tensor_type(torch.FloatTensor)


class GINGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, state_dim, hypernet_embed, weights_operation=None):
        super(GINGraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.state_dim = state_dim
        self.weights_operation = weights_operation

        self.hidden_features = int((in_features + out_features) / 2)

        # breaking the MLP to hypernetworks for deriving the weights and biases
        self.w1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                nn.ReLU(),
                                nn.Linear(hypernet_embed, in_features * self.hidden_features))
        self.b1 = nn.Linear(self.state_dim, self.hidden_features)

        self.w2 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                nn.ReLU(),
                                nn.Linear(hypernet_embed, self.hidden_features * out_features))
        self.b2 = nn.Linear(self.state_dim, out_features)

    def forward(self, input_features, adj, states):

        aggregated_input = torch.matmul(adj, input_features)

        batch_size = aggregated_input.size(0)

        w1 = self.w1(states).view(-1, self.in_features, self.hidden_features)
        w2 = self.w2(states).view(-1, self.hidden_features, self.out_features)

        if self.weights_operation == 'abs':
            w1 = torch.abs(w1)
            w2 = torch.abs(w2)
        elif self.weights_operation == 'clamp':
            w1 = nn.ReLU()(w1)
            w2 = nn.ReLU()(w2)
        elif self.weights_operation is None:
            pass
        else:
            raise NotImplementedError('The operation {} on the weights not implemented'.format(self.weights_operation))

        b1 = self.b1(states).view(batch_size, 1, -1).repeat(1, aggregated_input.size(1), 1)
        b2 = self.b2(states).view(batch_size, 1, -1).repeat(1, aggregated_input.size(1), 1)

        output1 = torch.nn.LeakyReLU()(torch.matmul(aggregated_input, w1) + b1)
        output = torch.matmul(output1, w2) + b2

        return output


class GNN(nn.Module):
    def __init__(self, num_input_features, hidden_layers, state_dim, hypernet_embed,
                 weights_operation=None, combine_type='gin'):
        super(GNN, self).__init__()

        self.num_input_features = num_input_features
        self.hidden_layers = hidden_layers
        self.state_dim = state_dim
        self.weights_operation = weights_operation

        self.nonlinearity = nn.ELU()

        self.layers = nn.ModuleList()
        for i in range(len(hidden_layers)):
            if i == 0:
                in_features = num_input_features
            else:
                in_features = hidden_layers[i - 1]
            out_features = hidden_layers[i]

            if combine_type == 'gin':
                self.layers.append(GINGraphConvolution(in_features, out_features, state_dim, hypernet_embed,
                                                       weights_operation=weights_operation))
            else:
                raise NotImplementedError('Layer type {} not supported!'.format(combine_type))

        # output layer parameters
        self.wout = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                  nn.ReLU(),
                                  nn.Linear(hypernet_embed, hidden_layers[-1]))
        self.wout_perNode = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                          nn.ReLU(),
                                          nn.Linear(hypernet_embed, hidden_layers[-1]))
        self.bout_perNode = nn.Sequential(nn.Linear(self.state_dim, hidden_layers[-1]),
                                          nn.ReLU(),
                                          nn.Linear(hidden_layers[-1], 1))

    def node_embedding(self, input_features, adj, states):
        x = input_features
        for i in range(len(self.layers)):
            x = self.nonlinearity(self.layers[i](x, adj, states))

        return x

    def forward(self, input_features, adj, states, num_agents):
        batch_size = input_features.size(0)
        output_features = self.node_embedding(input_features, adj,
                                              states)  # [:, :num_agents] # discard the outputs of entities
        readout, alive_agents_flags_1d = self.readout(output_features, adj, num_agents)

        wout = self.wout(states).view(batch_size, -1, 1)
        wout_perNode = self.wout_perNode(states).view(batch_size, -1, 1)
        bout_perNode = self.bout_perNode(states).view(batch_size, -1, 1).repeat(1, num_agents, 1)

        if self.weights_operation == 'abs':
            wout = torch.abs(wout)
        elif self.weights_operation == 'clamp':
            wout = nn.ReLU()(wout)
        elif self.weights_operation is None:
            pass
        else:
            raise NotImplementedError('The operation {} on the weights not implemented!'.format(self.weights_operation))

        scalar_out = torch.matmul(readout.view(batch_size, 1, -1), wout)
        per_node_scalars = torch.matmul(output_features.view(batch_size, num_agents, -1), wout_perNode) + bout_perNode
        per_node_scalars = nn.Softmax(dim=1)(per_node_scalars + (-1e10 * (1 - 1. * alive_agents_flags_1d)))

        return per_node_scalars, scalar_out

    def readout(self, x, adj, num_agents):
        alive_agents_flags_1d = (torch.max(adj, dim=2)[0] > 0).unsqueeze(2)
        alive_agents_flags = alive_agents_flags_1d.repeat(1, 1, x.size(2))

        # the constant in the denominator is to prevent "nan"s
        out = torch.sum(x * alive_agents_flags.float(), dim=1) / (torch.sum(alive_agents_flags, dim=1) + 1e-10)

        return out, alive_agents_flags_1d