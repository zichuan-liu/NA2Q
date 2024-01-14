import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class RecurrentTreeCell(nn.Module):
    def __init__(self, input_shape, hidden_shape, output_dim=1, tree_depth=3, beta=0, kernel='add'):
        super(RecurrentTreeCell, self).__init__()
        self.input_dim = input_shape
        self.hidden_shape = hidden_shape
        self.output_dim = output_dim
        self.tree_depth = tree_depth
        self.beta = beta
        self.max_leaf_idx = None  # the leaf index with maximal path probability
        self.kernel = kernel
        self._validate_parameters()

        self.q_tree_init()

    def q_tree_init(self):
        self.num_q_tree_nodes = 2 ** self.tree_depth - 1
        self.num_q_leaves = self.num_q_tree_nodes + 1  # Discretization?
        # self.hidden_shape = self.num_q_leaves

        self.q_leaves = nn.Parameter(torch.zeros([self.hidden_shape, self.num_q_leaves]),
                                     requires_grad=True)
        torch.nn.init.xavier_uniform_(self.q_leaves)

        # self.transform_q_dim = nn.Linear(self.hidden_shape, self.output_dim, bias=False)
        self.transform_q_dim = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_shape),
            nn.ELU(),
            nn.Linear(self.hidden_shape, self.hidden_shape*self.output_dim),
        )
        # torch.nn.init.xavier_uniform_(self.transform_q_dim.weight)

        self.q_logit_layers = []
        for cur_depth in range(self.tree_depth):
            self.q_logit_layers.append(nn.Linear(self.input_dim, self.hidden_shape * (2 ** cur_depth)))
            torch.nn.init.xavier_uniform_(self.q_logit_layers[-1].weight)
            self.q_logit_layers[-1].bias.data.fill_(self.beta)
        self.q_logit_layers = nn.ModuleList(self.q_logit_layers)

        if self.beta:
            self.betas_q = []
            for cur_depth in range(self.tree_depth):
                beta_array = np.full((self.hidden_shape, 2 ** cur_depth), self.beta)
                self.betas_q.append(nn.Parameter(torch.FloatTensor(beta_array), requires_grad=True))
            self.betas_q = nn.ParameterList(self.betas_q)


        self.h_logit_layers = []
        for cur_depth in range(self.tree_depth):
            self.h_logit_layers.append(nn.Linear(self.hidden_shape, self.hidden_shape * (2 ** cur_depth)))
            torch.nn.init.xavier_uniform_(self.h_logit_layers[-1].weight)
            self.h_logit_layers[-1].bias.data.fill_(self.beta)
        self.h_logit_layers = nn.ModuleList(self.h_logit_layers)

    def forward(self, obs, hidden_state):
        """
        Forward the tree for Q of each agent.
        Return the probabilities for reaching each leaf.
        """
        batch_size = obs.size()[0]
        obs = obs.view(batch_size, -1)
        hidden_state = hidden_state.view(-1, 1, self.hidden_shape)
        device = next(self.parameters()).device

        path_prob_qs = []
        path_prob_q = Variable(torch.ones(batch_size, self.hidden_shape, 1), requires_grad=True).to(device)
        path_prob_qs.append(path_prob_q)
        for cur_depth in range(self.tree_depth):
            # current_prob (bs, hidden_shape, 2**cur_depth)
            current_q_logit_left = self.q_logit_layers[cur_depth](obs).view(batch_size, self.hidden_shape, -1)
            current_h_logit_left = self.h_logit_layers[cur_depth](hidden_state) \
                .view(batch_size, self.hidden_shape, -1)

            if self.kernel == 'add':
                current_q_logit_left = current_q_logit_left+current_h_logit_left
            elif self.kernel == 'dot':
                current_q_logit_left = torch.einsum("blc, blc->blc", current_q_logit_left, current_h_logit_left)
            else:
                raise Exception("Error setting kernel, must 'add' or 'dot'.")

            if self.beta == 0:
                current_prob_left = torch.sigmoid(current_q_logit_left).view(batch_size, self.hidden_shape,
                                                                             2 ** cur_depth)
            else:
                current_prob_left = torch.sigmoid(
                    torch.einsum("blc, lc->blc", current_q_logit_left,
                                 self.betas_q[cur_depth])).view(batch_size, self.hidden_shape, 2 ** cur_depth)

            current_prob_right = 1 - current_prob_left

            current_prob_left = current_prob_left * path_prob_q
            current_prob_right = current_prob_right * path_prob_q

            # Update next level probability
            path_prob_q = torch.stack((current_prob_left.unsqueeze(-1), current_prob_right.unsqueeze(-1)), dim=3) \
                .view(batch_size, self.hidden_shape, 2 ** (cur_depth + 1))
            path_prob_qs.append(path_prob_q)

        _mu = torch.sum(path_prob_q, dim=1)
        vs, ids = torch.max(_mu, 1)  # ids is the leaf index with maximal path probability
        self.max_leaf_idx = ids
        # Sum the path probabilities of each layer
        # distribution_per_leaf = self.softmax(self.q_leaves)
        q_leaves = torch.einsum('bhd,hd->bh', path_prob_q, self.q_leaves)  # (bs, hidden_shape)
        # q = self.transform_q_dim(q_leaves)
        transform_weight = self.transform_q_dim(obs).view(batch_size, self.hidden_shape, self.output_dim)
        q_leaves = q_leaves.view(batch_size, 1, self.hidden_shape)
        q = torch.bmm(q_leaves, transform_weight).squeeze()

        return q, q_leaves, path_prob_qs

    def _validate_parameters(self):

        if not self.tree_depth > 0:
            msg = ("The Q tree depth should be strictly positive, but got {}"
                   "instead.")
            raise ValueError(msg.format(self.tree_depth))


class HistoryRTCs(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(HistoryRTCs, self).__init__()
        self.args = args

        self.depth = args.q_tree_depth

        self.q_tree = RecurrentTreeCell(input_shape, args.rnn_hidden_dim, args.n_actions,
                                   tree_depth=self.depth, beta=args.beta, kernel=args.kernel)
        self.path_prob_q = None
	
    def init_hidden(self):
        # make hidden states on same device as model
        device = next(self.parameters()).device
        return torch.zeros((1, self.args.rnn_hidden_dim)).to(device)
		
    def forward(self, obs, hidden_state):  # torch.Size([1, 96]) torch.Size([1, 64])

        h_in = hidden_state.view(-1, self.args.rnn_hidden_dim)

        # h_in = []
        # for agent in range(self.args.n_agents):
        #     h_in.append(hidden_state[:, :, agent].view(-1, self.args.rnn_hidden_dim))

        if self.args.evaluate:
            q, h, path_prob_q = self.q_tree(obs, h_in)
            self.path_prob_q = path_prob_q
        else:
            q, h, _ = self.q_tree(obs, h_in)
        # q = q.reshape(-1, self.args.n_actions, self.args.central_action_embed)
        return q, h
