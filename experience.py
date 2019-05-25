from typing import Tuple

import torch


class ExperienceBatch:

    def __init__(self,
                 observations,
                 actions,
                 prev_actions,
                 action_log_probs,
                 returns,
                 value_predictions,
                 advantage_targets,
                 masks,
                 recurrent_hidden_states):
        num_steps, num_envs = actions.shape[:2]
        self.observations = self._flatten(observations, num_steps, num_envs)
        self.actions = self._flatten(actions, num_steps, num_envs)
        self.prev_actions = self._flatten(prev_actions, num_steps, num_envs)
        self.action_log_probs = self._flatten(action_log_probs, num_steps, num_envs)
        self.returns = self._flatten(returns, num_steps, num_envs)
        self.value_predictions = self._flatten(value_predictions, num_steps, num_envs)
        self.advantage_targets = self._flatten(advantage_targets, num_steps, num_envs)
        self.masks = self._flatten(masks, num_steps, num_envs)
        self.recurrent_hidden_states = recurrent_hidden_states.view(num_envs, -1)

    def action_eval_input(self):
        return (self.observations,
                self.recurrent_hidden_states,
                self.masks,
                self.prev_actions,
                self.actions)

    def _flatten(self, tensor, num_steps, num_envs):
        return tensor.view(num_steps * num_envs, *tensor.shape[2:])


class ExperienceStorage:

    def __init__(self,
                 num_steps: int,
                 num_envs: int,
                 observation_shape: Tuple,
                 recurrent_hidden_size: int,
                 device: torch.device):
        self._num_steps = num_steps
        self._num_envs = num_envs
        self._step = 0
        self._device = device

        self.observations = torch.zeros(num_steps + 1,
                                        num_envs,
                                        *observation_shape,
                                        dtype=torch.uint8).to(device)
        self.actions = torch.zeros(num_steps, num_envs, 1, dtype=torch.long).to(device)
        self.action_log_probs = torch.zeros(num_steps, num_envs, 1).to(device)
        self.rewards = torch.zeros(num_steps, num_envs, 1).to(device)
        self.value_predictions = torch.zeros(num_steps + 1, num_envs, 1).to(device)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1).to(device)
        self.masks = torch.ones(num_steps + 1, num_envs, 1).to(device)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_envs, recurrent_hidden_size).to(device)

    def insert(self,
               observations,
               actions,
               action_log_probs,
               rewards,
               value_predictions,
               masks,
               recurrent_hidden_states):
        self.observations[self._step + 1].copy_(observations)
        self.actions[self._step].copy_(actions)
        self.action_log_probs[self._step].copy_(action_log_probs)
        self.rewards[self._step].copy_(rewards)
        self.value_predictions[self._step].copy_(value_predictions)
        self.masks[self._step + 1].copy_(masks)
        self.recurrent_hidden_states[self._step + 1].copy_(recurrent_hidden_states)
        self._step = (self._step + 1) % self._num_steps

    def insert_initial_observations(self, observations):
        self.observations[0].copy_(observations)

    def get_actor_input(self, step):
        states = self.observations[step]
        rnn_hxs = self.recurrent_hidden_states[step]
        masks = self.masks[step]
        prev_actions = self.get_prev_actions(step)
        return states, rnn_hxs, masks, prev_actions

    def get_prev_actions(self, step, last_n=4):
        prev_action_indices = [step - i for i in range(1, last_n + 1)]
        prev_actions = self.actions[prev_action_indices, :].permute(1, 0, 2)
        return prev_actions

    def get_critic_input(self):
        return self.get_actor_input(step=-1)

    def compute_gae_returns(self, next_value, gamma, gae_lambda):
        self.value_predictions[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + \
                gamma * self.value_predictions[step + 1] * self.masks[step + 1] - \
                self.value_predictions[step]
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_predictions[step]

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_advantages(self, eps: float = 1e-5):
        advantages = self.returns[:-1] - self.value_predictions[:-1]
        norm_advantages = (advantages - advantages.mean()) / (advantages.std() + eps)
        return norm_advantages

    def batches(self, advantages: torch.tensor, minibatches: int):
        """Yield experience batches for recurrent policy training."""
        assert (self._num_envs % minibatches) == 0
        num_envs_per_batch = self._num_envs // minibatches
        random_env_indices = torch.randperm(self._num_envs)

        for start in range(0, self._num_envs, num_envs_per_batch):
            end = start + num_envs_per_batch
            indices = random_env_indices[start:end]

            prev_actions_shape = (self._num_steps, num_envs_per_batch, 4, 1)
            prev_actions = torch.zeros(*prev_actions_shape,
                                       dtype=torch.long).to(self._device)

            for step in range(self._num_steps):
                actions = self.get_prev_actions(step)
                prev_actions[step, :] = actions[indices]

            yield ExperienceBatch(
                observations=self.observations[:-1, indices],
                actions=self.actions[:, indices],
                prev_actions=prev_actions,
                action_log_probs=self.action_log_probs[:, indices],
                returns=self.returns[:-1, indices],
                value_predictions=self.value_predictions[:-1, indices],
                advantage_targets=advantages[:, indices],
                masks=self.masks[:-1, indices],
                recurrent_hidden_states=self.recurrent_hidden_states[:1, indices]
            )
