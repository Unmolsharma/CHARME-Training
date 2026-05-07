"""PPO for CHARME.

Source: authors' exact code from charme-rl repo.
Fixes applied to update():
  1. Removed references to notebook globals
  2. Buffer is self.buffer (in-memory) instead of loading from disk files
  3. evaluate() now called with (state_dict, actions) instead of positional tensors
  4. Advantage-weighted sampling from authors' logic preserved exactly
  5. Logical attr/edge_index padded to handle variable graph sizes
Everything else (RolloutBuffer, select_action, save, load) is unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .models import ActorCritic


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.actions_minor = []
        self.states = {'hw_attr': [], 'emb_matrix': [], 'logical_index': []}
        self.logprobs = []
        self.rewards = []
        self.rewards_distance = []
        self.state_values = []
        self.is_terminals = []

    def clear_top_20percent(self, memory_size):
        clear_size = int(0.2 * memory_size)
        del self.states['hw_attr'][:clear_size]
        del self.states['emb_matrix'][:clear_size]
        del self.states['logical_index'][:clear_size]
        del self.actions[:clear_size]
        del self.logprobs[:clear_size]
        del self.rewards[:clear_size]
        del self.rewards_distance[:clear_size]
        del self.state_values[:clear_size]
        del self.is_terminals[:clear_size]

    def clear_partial(self, memory_size):
        clear_size = len(self.rewards) - memory_size
        del self.states['hw_attr'][clear_size:]
        del self.states['emb_matrix'][clear_size:]
        del self.states['logical_index'][clear_size:]
        del self.actions[clear_size:]
        del self.logprobs[clear_size:]
        del self.rewards[clear_size:]
        del self.rewards_distance[clear_size:]
        del self.state_values[clear_size:]
        del self.is_terminals[clear_size:]

    def clear_all(self):
        clear_size = len(self.rewards)
        del self.states['hw_attr'][:clear_size]
        del self.states['emb_matrix'][:clear_size]
        del self.states['logical_index'][:clear_size]
        del self.actions[:clear_size]
        del self.logprobs[:clear_size]
        del self.rewards[:clear_size]
        del self.rewards_distance[:clear_size]
        del self.state_values[:clear_size]
        del self.is_terminals[:clear_size]

    def len(self):
        return len(self.rewards)


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma,
                 K_epochs, eps_clip, has_continuous_action_space,
                 action_std_init=0.6, device=None):
        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.max_prob = 0
        self.device = device

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(
            state_dim, action_dim, has_continuous_action_space,
            action_std_init, device
        )
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.policy_old = ActorCritic(
            state_dim, action_dim, has_continuous_action_space,
            action_std_init, device
        )
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

    def select_action(self, state, mask, mask_connected):
        with torch.no_grad():
            pred_action, action_logprob, state_val = self.policy_old.act(
                state, mask, mask_connected
            )
            self.max_prob = self.policy_old.max_prob
        return state, pred_action, action_logprob, state_val

    def _pad_stack(self, tensors, dim=0):
        """Stack tensors of different sizes by padding to max size along dim."""
        max_size = max(t.shape[dim] for t in tensors)
        padded = []
        for t in tensors:
            pad_size = max_size - t.shape[dim]
            if pad_size > 0:
                if dim == 0:
                    t = torch.nn.functional.pad(t, (0, 0, 0, pad_size))
                else:
                    t = torch.nn.functional.pad(t, (0, pad_size))
            padded.append(t)
        return torch.stack(padded)

    def update(self, env):
        device = self.device
        buffer = self.buffer

        if buffer.len() == 0:
            return

        # Monte Carlo returns
        rewards = []
        discounted_reward = 0
        for reward, reward_distance, is_terminal in zip(
            reversed(buffer.rewards),
            reversed(buffer.rewards_distance),
            reversed(buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward + reward_distance)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Advantage-weighted index selection
        old_state_values = torch.squeeze(
            torch.stack(buffer.state_values, dim=0)
        ).detach().to(device)

        advantages = rewards.detach() - old_state_values.detach()
        adv_np = advantages.detach().cpu().numpy()
        adv_np[adv_np < 0] = 0
        nums_nonzero = np.count_nonzero(adv_np)

        if nums_nonzero == 0:
            buffer.clear_all()
            self.policy_old.load_state_dict(self.policy.state_dict())
            return

        adv_np = adv_np / adv_np.sum()
        possible_index = np.array(range(0, buffer.len()))

        selected_index = np.random.choice(
            possible_index,
            size=int(min(nums_nonzero, buffer.len())),
            replace=False,
            p=adv_np
        )

        advantages = advantages[selected_index]
        rewards_full = rewards.clone()
        rewards = rewards[selected_index]

        # Reconstruct state tensors — pad to handle variable graph sizes
        logical_attrs = [
            env.fix_state_list[buffer.states['logical_index'][i]]['logical_attr'].clone()
            for i in selected_index
        ]
        logical_attr = self._pad_stack(logical_attrs, dim=0).detach().to(device)

        logical_attrs_full = [
            env.fix_state_list[buffer.states['logical_index'][i]]['logical_attr'].clone()
            for i in possible_index
        ]
        logical_attr_full = self._pad_stack(logical_attrs_full, dim=0).detach().to(device)

        lei = [
            env.fix_state_list[buffer.states['logical_index'][i]]['logical_edge_index'].clone()
            for i in selected_index
        ]
        max_e = max(t.shape[1] for t in lei)
        logical_edge_index = torch.stack([
            torch.nn.functional.pad(t, (0, max_e - t.shape[1])) for t in lei
        ]).detach().to(device)

        lei_f = [
            env.fix_state_list[buffer.states['logical_index'][i]]['logical_edge_index'].clone()
            for i in possible_index
        ]
        max_e_f = max(t.shape[1] for t in lei_f)
        logical_edge_index_full = torch.stack([
            torch.nn.functional.pad(t, (0, max_e_f - t.shape[1])) for t in lei_f
        ]).detach().to(device)

        hw_attr = torch.stack([
            buffer.states['hw_attr'][i] for i in selected_index
        ]).detach().to(device)

        hw_attr_full = torch.stack([
            buffer.states['hw_attr'][i] for i in possible_index
        ]).detach().to(device)

        hw_edge_index = torch.stack([
            env.fix_state_list[buffer.states['logical_index'][i]]['hw_edge_index'].clone()
            for i in selected_index
        ]).detach().to(device)

        hw_edge_index_full = torch.stack([
            env.fix_state_list[buffer.states['logical_index'][i]]['hw_edge_index'].clone()
            for i in possible_index
        ]).detach().to(device)

        emb_matrix = torch.stack([
            buffer.states['emb_matrix'][i] for i in selected_index
        ], dim=0).detach().to(device)

        emb_matrix_full = torch.stack([
            buffer.states['emb_matrix'][i] for i in possible_index
        ], dim=0).detach().to(device)

        old_states = {
            'logical_attr': logical_attr,
            'logical_edge_index': logical_edge_index,
            'hw_attr': hw_attr,
            'hw_edge_index': hw_edge_index,
            'emb_matrix': emb_matrix,
        }
        old_states_full = {
            'logical_attr': logical_attr_full,
            'logical_edge_index': logical_edge_index_full,
            'hw_attr': hw_attr_full,
            'hw_edge_index': hw_edge_index_full,
            'emb_matrix': emb_matrix_full,
        }

        old_actions = torch.squeeze(
            torch.stack(buffer.actions, dim=0)[selected_index]
        ).detach().to(device)

        old_logprobs = torch.squeeze(
            torch.stack(buffer.logprobs, dim=0)[selected_index]
        ).detach().to(device)

        self.eps_clip = 0.2

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            state_values_full = self.policy.critic(old_states_full)
            state_values_full = torch.squeeze(state_values_full)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(
                ratios, 1 - self.eps_clip, 1 + self.eps_clip
            ) * advantages

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values_full, rewards_full)
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        del rewards
        del logical_attr
        del logical_edge_index
        del hw_attr
        del hw_edge_index
        del emb_matrix
        del old_actions
        del old_logprobs
        del old_state_values
        buffer.clear_all()

        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        ))
        self.policy.load_state_dict(torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        ))
