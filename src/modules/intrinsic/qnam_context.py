import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions import kl_divergence
import argparse


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, h_dim, state_dim, hidden_dim=32):
        super(VAE, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = h_dim
        self.state_dim = state_dim
        self.h_dim = h_dim

        self.e1 = nn.Linear(h_dim, self.hidden_dim)
        self.e2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.mean = nn.Linear(self.hidden_dim, self.latent_dim)
        self.log_std = nn.Linear(self.hidden_dim, self.latent_dim)

        self.d1 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.d2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.d3 = nn.Linear(self.hidden_dim, state_dim)

    def forward(self, latent):
        z, mean, std = self.encode(latent)

        u = self.decode(z)

        u = torch.sigmoid(u)
        return u, mean, std

    def encode(self, latent):
        z = F.relu(self.e1(latent))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z)#.clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        return z, mean, std

    def decode(self, z):
        s = F.relu(self.d1(z))
        s = F.relu(self.d2(s))
        return self.d3(s)

