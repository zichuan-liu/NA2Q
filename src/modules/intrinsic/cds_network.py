import torch
import torch.optim as optim
from torch import nn as nn
from torch.nn import functional as F


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class LayerNorm(nn.Module):
    """
    Simple 1D LayerNorm.
    """

    def __init__(self, features, center=True, scale=False, eps=1e-6):
        super().__init__()
        self.center = center
        self.scale = scale
        self.eps = eps
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(features))
        else:
            self.scale_param = None
        if self.center:
            self.center_param = nn.Parameter(torch.zeros(features))
        else:
            self.center_param = None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = (x - mean) / (std + self.eps)
        if self.scale:
            output = output * self.scale_param
        if self.center:
            output = output + self.center_param
        return


class Predict_Network1(nn.Module):

    def __init__(self, num_inputs, hidden_dim, num_outputs, layer_norm=True, lr=1e-3):
        super(Predict_Network1, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, num_outputs)

        self.layer_norm = layer_norm
        if layer_norm:
            self.ln1 = LayerNorm(hidden_dim)

        self.apply(weights_init_)
        self.lr = lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input):
        if self.layer_norm:
            h = F.relu(self.ln1(self.linear1(input)))
        else:
            h = F.relu(self.linear1(input))

        h = F.relu(self.linear2(h))
        x = self.last_fc(h)
        return x

    def get_log_pi(self, own_variable, other_variable):
        predict_variable = self.forward(own_variable)
        log_prob = -1 * F.mse_loss(predict_variable, other_variable, reduction='none')
        log_prob = torch.sum(log_prob, -1, keepdim=True)
        return log_prob

    def update(self, own_variable, other_variable, mask):
        predict_variable = self.forward(own_variable)
        loss = F.mse_loss(predict_variable, other_variable, reduction='none')
        loss = loss.sum(dim=-1, keepdim=True)
        loss = (loss * mask).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optimizer.step()


class Predict_Network1_combine(nn.Module):

    def __init__(self, num_inputs, hidden_dim, num_outputs, n_agents, layer_norm=True, lr=1e-3):
        super(Predict_Network1_combine, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim + n_agents, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, num_outputs)

        self.layer_norm = layer_norm
        if layer_norm:
            self.ln1 = LayerNorm(hidden_dim)

        self.apply(weights_init_)
        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input, add_id):
        if self.layer_norm:
            h = F.relu(self.ln1(self.linear1(input)))
        else:
            h = F.relu(self.linear1(input))

        h = torch.cat([h, add_id], dim=-1)
        h = F.relu(self.linear2(h))
        x = self.last_fc(h)
        return x

    def get_log_pi(self, own_variable, other_variable, add_id):
        predict_variable = self.forward(own_variable, add_id)
        log_prob = -1 * F.mse_loss(predict_variable, other_variable, reduction='none')
        log_prob = torch.sum(log_prob, -1, keepdim=True)
        return log_prob

    def update(self, own_variable, other_variable, add_id, mask):
        predict_variable = self.forward(own_variable, add_id)
        loss = F.mse_loss(predict_variable, other_variable, reduction='none')
        loss = loss.sum(dim=-1, keepdim=True)
        loss = (loss * mask).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optimizer.step()