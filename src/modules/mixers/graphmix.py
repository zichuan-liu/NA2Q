import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphMixer(nn.Module):
    def __init__(self, args):
        super(GraphMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.obs_dim = args.obs_shape
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.embed_dim = args.mixing_embed_dim
        hypernet_embed = self.args.hypernet_embed

        # mixing GNN
        combine_type = 'gin'
        self.mixing_GNN = GNN(num_input_features=1, hidden_layers=[self.embed_dim],
                              state_dim=self.state_dim, hypernet_embed=hypernet_embed,
                              weights_operation='abs',
                              combine_type=combine_type)

        # attention mechanism
        self.enc_obs = True
        obs_dim = self.rnn_hidden_dim
        if self.enc_obs:
            self.obs_enc_dim = 16
            self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, self.obs_enc_dim),
                                             nn.ReLU())
            self.obs_dim_effective = self.obs_enc_dim
        else:
            self.obs_encoder = nn.Sequential()
            self.obs_dim_effective = obs_dim

        self.state_encoder = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))

        self.W_attn_query = nn.Linear(self.obs_dim_effective, self.obs_dim_effective, bias=False)
        self.W_attn_key = nn.Linear(self.obs_dim_effective, self.obs_dim_effective, bias=False)

        # output bias
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states,
                agent_obs=None,
                team_rewards=None,
                hidden_states=None):

        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        emb_states = th.abs(self.state_encoder(states)).view(-1, self.n_agents, self.embed_dim)

        agent_qs = agent_qs.view(-1, self.n_agents, 1)

        # find the agents which are alive
        alive_agents = 1. * (th.sum(agent_obs, dim=3) > 0).view(-1, self.n_agents)
        # create a mask for isolating nodes which are dead by taking the outer product of the above tensor with itself
        alive_agents_mask = th.bmm(alive_agents.unsqueeze(2), alive_agents.unsqueeze(1))

        # encode hidden states
        encoded_hidden_states = self.obs_encoder(hidden_states)
        encoded_hidden_states = encoded_hidden_states.contiguous().view(-1, self.n_agents, self.obs_dim_effective)

        # GAT
        # adjacency based on the attention mechanism
        attn_query = self.W_attn_query(encoded_hidden_states)
        attn_key = self.W_attn_key(encoded_hidden_states)
        attn = th.matmul(attn_query, th.transpose(attn_key, 1, 2)) / np.sqrt(self.obs_dim_effective)

        # make the attention with softmax very small for dead agents so they get zero attention
        attn = nn.Softmax(dim=2)(attn + (-1e10 * (1 - alive_agents_mask)))
        batch_adj = attn * alive_agents_mask  # completely isolate the dead agents in the graph

        # print(batch_adj.shape, states.shape, agent_qs.shape)
        #torch.Size([2240, 5, 5]) torch.Size([2240, 5, 32]) torch.Size([2240, 5, 1])

        K = th.mean(th.bmm(batch_adj, emb_states), dim=-1)#torch.Size([2240, 5])

        K = K.view(-1, 1, self.n_agents)

        if self.args.is_gnn:
		    _, K = self.mixing_GNN(encoded_hidden_states, batch_adj, states, self.n_agents)
        y = th.bmm(K, agent_qs)
        # state-dependent bias
        v = self.V(states).view(-1, 1, 1)
        q_tot = (y + v).view(bs, -1, 1)

        return q_tot