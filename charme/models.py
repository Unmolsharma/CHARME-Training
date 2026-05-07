"""Neural network models for CHARME.

Source: authors' exact code from charme-rl repo.
Only fix applied: evaluate() signature corrected to accept state dict
instead of positional tensor args (was broken in original).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal, Categorical

try:
    from torch_geometric.nn import GCNConv
except Exception:
    GCNConv = None


class critic(nn.Module):
    def __init__(self, in_logical_channels, hidden_logical_channels,
                 in_hardware_channels, hidden_hardware_channels,
                 out_channels, logical_size, device):
        super().__init__()
        if GCNConv is None:
            raise ImportError("torch_geometric is required. pip install torch-geometric")

        # Authors' exact layer names preserved
        self.conv_logical_1 = GCNConv(in_logical_channels, hidden_logical_channels)
        self.conv_logical_2 = GCNConv(hidden_logical_channels, hidden_logical_channels)
        self.conv_logical_3 = GCNConv(hidden_logical_channels, hidden_logical_channels)

        self.conv_hardware_1 = GCNConv(in_hardware_channels, hidden_hardware_channels)
        self.conv_hardware_2 = GCNConv(hidden_hardware_channels, hidden_hardware_channels)
        self.conv_hardware_3 = GCNConv(hidden_hardware_channels, hidden_hardware_channels)

        # 128 = hidden_logical(64) + hidden_hardware(64) — authors' exact hardcoded value
        self.lin = nn.Linear(128, 1)
        self.logical_size = logical_size
        self.device = device

    def forward(self, state):
        # Authors' exact forward — unchanged
        x_logical = state['logical_attr']
        logical_edge_index = state['logical_edge_index']
        x_hardware = state['hw_attr']
        hardware_edge_index = state['hw_edge_index']
        embedding_matrix = state['emb_matrix']

        if x_logical.dim() == 2:
            x_logical = x_logical.unsqueeze(dim=0)
        if logical_edge_index.dim() == 2:
            logical_edge_index = logical_edge_index.unsqueeze(dim=0)
        if x_hardware.dim() == 2:
            x_hardware = x_hardware.unsqueeze(dim=0)
        if hardware_edge_index.dim() == 2:
            hardware_edge_index = hardware_edge_index.unsqueeze(dim=0)
        if embedding_matrix.dim() == 2:
            embedding_matrix = embedding_matrix.unsqueeze(dim=0)

        minibatch_size = x_logical.shape[0]

        x_logical_next = [
            self.conv_logical_3(
                self.conv_logical_2(
                    self.conv_logical_1(x_logical[i], logical_edge_index[i]).to(self.device),
                    logical_edge_index[i]
                ).to(self.device),
                logical_edge_index[i]
            ).to(self.device).unsqueeze(dim=0)
            for i in range(minibatch_size)
        ]
        x_logical_next = torch.cat(x_logical_next, dim=0)

        x_hardware_next = [
            self.conv_hardware_3(
                self.conv_hardware_2(
                    self.conv_hardware_1(x_hardware[i], hardware_edge_index[i]).to(self.device),
                    hardware_edge_index[i]
                ).to(self.device),
                hardware_edge_index[i]
            ).to(self.device).unsqueeze(dim=0)
            for i in range(minibatch_size)
        ]
        x_hardware_next = torch.cat(x_hardware_next, dim=0)

        # Equation 12: Y_aggr = M_phi^T @ Y_H
        x_emb = torch.stack([
            torch.sparse.mm(embedding_matrix[i], x_hardware_next[i])
            for i in range(minibatch_size)
        ])

        x_final = torch.cat((x_logical_next, x_emb), dim=-1)
        y = self.lin(x_final)  # (B, N, 1)
        y = y.mean(dim=1)  # global mean pool -> (B, 1)
        return y.squeeze(-1)  # (B,)


class actor(nn.Module):
    def __init__(self, in_logical_channels, hidden_logical_channels,
                 in_hardware_channels, hidden_hardware_channels,
                 out_channels, logical_size, action_dim, device):
        super().__init__()
        if GCNConv is None:
            raise ImportError("torch_geometric is required. pip install torch-geometric")

        self.conv_logical_1 = GCNConv(in_logical_channels, hidden_logical_channels)
        self.conv_logical_2 = GCNConv(hidden_logical_channels, hidden_logical_channels)
        self.conv_logical_3 = GCNConv(hidden_logical_channels, hidden_logical_channels)

        self.conv_hardware_1 = GCNConv(in_hardware_channels, hidden_hardware_channels)
        self.conv_hardware_2 = GCNConv(hidden_hardware_channels, hidden_hardware_channels)
        self.conv_hardware_3 = GCNConv(hidden_hardware_channels, hidden_hardware_channels)

        self.lin = nn.Linear(128, 1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.logical_size = logical_size
        self.action_dim = action_dim
        self.device = device

    def forward(self, state):
        # Authors' exact forward — unchanged
        x_logical = state['logical_attr']
        logical_edge_index = state['logical_edge_index']
        x_hardware = state['hw_attr']
        hardware_edge_index = state['hw_edge_index']
        embedding_matrix = state['emb_matrix']

        if x_logical.dim() == 2:
            x_logical = x_logical.unsqueeze(dim=0)
        if logical_edge_index.dim() == 2:
            logical_edge_index = logical_edge_index.unsqueeze(dim=0)
        if x_hardware.dim() == 2:
            x_hardware = x_hardware.unsqueeze(dim=0)
        if hardware_edge_index.dim() == 2:
            hardware_edge_index = hardware_edge_index.unsqueeze(dim=0)
        if embedding_matrix.dim() == 2:
            embedding_matrix = embedding_matrix.unsqueeze(dim=0)

        minibatch_size = x_logical.shape[0]

        x_logical_next = [
            self.conv_logical_3(
                self.conv_logical_2(
                    self.conv_logical_1(x_logical[i], logical_edge_index[i]).to(self.device),
                    logical_edge_index[i]
                ).to(self.device),
                logical_edge_index[i]
            ).to(self.device).unsqueeze(dim=0)
            for i in range(minibatch_size)
        ]
        x_logical_next = torch.cat(x_logical_next, dim=0)

        x_hardware_next = [
            self.conv_hardware_3(
                self.conv_hardware_2(
                    self.conv_hardware_1(x_hardware[i], hardware_edge_index[i]).to(self.device),
                    hardware_edge_index[i]
                ).to(self.device),
                hardware_edge_index[i]
            ).to(self.device).unsqueeze(dim=0)
            for i in range(minibatch_size)
        ]
        x_hardware_next = torch.cat(x_hardware_next, dim=0)

        x_emb = torch.stack([
            torch.sparse.mm(embedding_matrix[i], x_hardware_next[i])
            for i in range(minibatch_size)
        ])

        x_final = torch.cat((x_logical_next, x_emb), dim=-1)
        y = self.softmax(self.lin(x_final).squeeze(dim=-1))
        return y


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space,
                 action_std_init, device,
                 in_logical_channels=1, hidden_logical_channels=64,
                 in_hardware_channels=1, hidden_hardware_channels=64):
        super().__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim
        self.device = device
        self.max_prob = torch.tensor(0).to(self.device)

        if has_continuous_action_space:
            self.action_var = torch.full(
                (action_dim,), action_std_init * action_std_init
            ).to(device)

        self.actor = actor(
            in_logical_channels, hidden_logical_channels,
            in_hardware_channels, hidden_hardware_channels,
            out_channels=hidden_logical_channels,
            logical_size=action_dim, action_dim=action_dim,
            device=device
        ).to(device)

        self.critic = critic(
            in_logical_channels, hidden_logical_channels,
            in_hardware_channels, hidden_hardware_channels,
            out_channels=hidden_logical_channels,
            logical_size=action_dim,
            device=device
        ).to(device)

    def set_action_std(self, new_action_std: float):
        if self.has_continuous_action_space:
            self.action_var = torch.full(
                (self.action_dim,), new_action_std * new_action_std
            ).to(self.device)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def act(self, state, mask, mask_connected):
        # Authors' exact act() — unchanged
        state_cuda = {}
        for k in state.keys():
            if k != 'emb_dict':
                state_cuda[k] = state[k].to(self.device)
            else:
                state_cuda[k] = state[k]

        action_prob = self.actor(state_cuda).squeeze()
        mask_len = len(mask)
        prob_len = action_prob.shape[-1]
        if prob_len < mask_len:
            padding = torch.zeros(mask_len - prob_len, device=self.device)
            action_prob = torch.cat([action_prob, padding])
        elif prob_len > mask_len:
            action_prob = action_prob[:mask_len]
        action_prob = torch.add(action_prob, 0.000000000001)

        final_mask = [a or b for a, b in zip(mask, mask_connected)]
        mask_tensor = torch.tensor(final_mask, device=self.device)
        action_prob = action_prob.masked_fill(mask_tensor == True, 0)

        dist = Categorical(action_prob)
        pred_action = dist.sample()
        action_logprob = dist.log_prob(pred_action)
        state_val = self.critic(state_cuda).squeeze()

        return pred_action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        """
        FIX: Original evaluate() called self.actor() with positional tensor args
        (broken — actor.forward expects a state dict). Fixed to pass state dict.
        The rest of the logic is preserved from the authors' version.
        """
        action_probs = self.actor(state)  # state is a dict
        action_probs = action_probs.squeeze()

        if self.has_continuous_action_space:
            action_var = self.action_var.expand_as(action_probs)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_probs, cov_mat)
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
        else:
            dist = Categorical(action_probs)
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()

        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy
