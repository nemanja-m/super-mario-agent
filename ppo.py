from collections import defaultdict

import torch
import torch.nn as nn

from experience import ExperienceStorage
from policy import RecurrentPolicy


class PPOAgent:

    def __init__(self,
                 actor_critic: RecurrentPolicy,
                 lr: float = 7e-4,
                 clip_threshold: float = 0.2,
                 epochs: int = 4,
                 minibatches: int = 2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5):
        self._actor_critic = actor_critic
        self._clip_threshold = clip_threshold
        self._epochs = epochs
        self._minibatches = minibatches
        self._value_loss_coef = value_loss_coef
        self._entropy_coef = entropy_coef
        self._max_grad_norm = max_grad_norm
        self._optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)

    def update(self, experience_storage: ExperienceStorage):
        losses = defaultdict(int)

        for epoch in range(self._epochs):
            for exp_batch in experience_storage.batches(self._minibatches):
                eval_input = exp_batch.action_eval_input()
                (values,
                 action_log_probs,
                 action_dist_entropy) = self._actor_critic.evaluate_actions(*eval_input)

                policy_loss = self._policy_loss(action_log_probs,
                                                exp_batch.action_log_probs,
                                                exp_batch.advantage_targets)
                value_loss = self._value_loss(values,
                                              exp_batch.value_predictions,
                                              exp_batch.returns)
                loss = policy_loss + \
                    self._value_loss_coef * value_loss - \
                    self._entropy_coef * action_dist_entropy

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

    def _value_loss(self, values, value_predictions, returns):
        value_preds_clipped = value_predictions + \
            (values - value_predictions).clamp(-self._clip_threshold,
                                               self._clip_threshold)
        value_losses = (values - returns).pow(2)
        value_losses_clipped = (value_preds_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        return value_loss
