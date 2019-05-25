from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable

import torch
import torch.nn as nn

from experience import ExperienceStorage
from policy import RecurrentPolicy


EPS = 1e-5


class Agent(ABC):

    def __init__(self,
                 actor_critic: RecurrentPolicy,
                 lr: float,
                 lr_lambda: Callable[[int], float],
                 policy_loss_coef: float,
                 value_loss_coef: float,
                 entropy_loss_coef: float,
                 max_grad_norm: float):
        self._actor_critic = actor_critic
        self._policy_loss_coef = policy_loss_coef
        self._value_loss_coef = value_loss_coef
        self._entropy_loss_coef = entropy_loss_coef
        self._max_grad_norm = max_grad_norm
        self._optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr, eps=EPS)
        self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer,
                                                               lr_lambda)

    def current_lr(self):
        [lr] = self._lr_scheduler.get_lr()
        return lr

    @abstractmethod
    def update(self, experience_storage: ExperienceStorage):
        pass


class PPOAgent(Agent):

    def __init__(self,
                 *args,
                 **kwargs):
        self._clip_threshold = kwargs.pop('clip_threshold')
        self._epochs = kwargs.pop('epochs')
        self._minibatches = kwargs.pop('minibatches')
        super().__init__(*args, **kwargs)

    def update(self, experience_storage: ExperienceStorage):
        losses = defaultdict(int)
        advantages = experience_storage.compute_advantages()

        self._lr_scheduler.step()

        for epoch in range(self._epochs):
            for exp_batch in experience_storage.batches(advantages, self._minibatches):
                eval_input = exp_batch.action_eval_input()
                (values,
                 action_log_probs,
                 action_dist_entropy) = self._actor_critic.evaluate_actions(*eval_input)

                policy_loss = self._policy_loss(action_log_probs,
                                                exp_batch.action_log_probs,
                                                exp_batch.advantage_targets)
                value_loss = 0.5 * (exp_batch.returns - values).pow(2).mean()

                loss = self._policy_loss_coef * policy_loss + \
                    self._value_loss_coef * value_loss - \
                    self._entropy_loss_coef * action_dist_entropy

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._actor_critic.parameters(),
                                         self._max_grad_norm)
                self._optimizer.step()

                losses['value_loss'] += value_loss.item()
                losses['policy_loss'] += policy_loss.item()
                losses['action_dist_entropy'] += action_dist_entropy.item()

        num_updates = self._epochs * self._minibatches
        for loss_name in losses.keys():
            losses[loss_name] /= num_updates

        experience_storage.after_update()
        return losses

    def _policy_loss(self, action_log_probs, old_action_log_probs, advantage_targets):
        ratio = torch.exp(action_log_probs - old_action_log_probs)
        ratio_term = ratio * advantage_targets
        clamp = torch.clamp(ratio,
                            1 - self._clip_threshold,
                            1 + self._clip_threshold)
        clamp_term = clamp * advantage_targets
        policy_loss = -torch.min(ratio_term, clamp_term).mean()
        return policy_loss
