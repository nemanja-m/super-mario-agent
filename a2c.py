from collections import defaultdict
from typing import Callable

import torch
import torch.nn as nn

from experience import ExperienceStorage
from policy import BasePolicy


_START_LR = 1.0e-4


class A2CAgent:

    def __init__(self,
                 actor_critic: BasePolicy,
                 lr: float = _START_LR,
                 lr_lambda: Callable[[int], float] = lambda step: _START_LR,
                 clip_threshold: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.001,
                 max_grad_norm: float = 0.5):
        self._actor_critic = actor_critic
        self._clip_threshold = clip_threshold
        self._value_loss_coef = value_loss_coef
        self._entropy_coef = entropy_coef
        self._max_grad_norm = max_grad_norm
        self._optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr, eps=1e-5)
        self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer,
                                                               lr_lambda)

    def current_lr(self):
        [lr] = self._lr_scheduler.get_lr()
        return lr

    def update(self, experience_storage: ExperienceStorage):
        losses = defaultdict(int)

        obs_shape = experience_storage.observations.size()[2:]
        action_shape = experience_storage.actions.size()[-1]
        num_steps, num_processes, _ = experience_storage.rewards.size()

        eval_input = (
            experience_storage.observations[:-1].view(-1, *obs_shape),
            experience_storage.recurrent_hidden_states[0].view(
                -1, self._actor_critic._recurrent_hidden_size),
            experience_storage.masks[:-1].view(-1, 1),
            None,
            experience_storage.actions.view(-1, action_shape)
        )

        (values,
         action_log_probs,
         action_dist_entropy) = self._actor_critic.evaluate_actions(*eval_input)

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = experience_storage.returns[:-1] - values
        policy_loss = -(advantages.detach() * action_log_probs).mean()
        value_loss = 0.5 * advantages.pow(2).mean()

        loss = policy_loss + \
            self._value_loss_coef * value_loss - \
            self._entropy_coef * action_dist_entropy

        self._lr_scheduler.step()
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._actor_critic.parameters(),
                                 self._max_grad_norm)
        self._optimizer.step()

        losses['value_loss'] += value_loss.item()
        losses['policy_loss'] += policy_loss.item()
        losses['action_dist_entropy'] += action_dist_entropy.item()
        return losses
