import torch
from typing import Tuple


class ExperienceStorage:

    def __init__(self,
                 num_steps: int,
                 num_envs: int,
                 observation_shape: Tuple,
                 recurrent_hidden_state_size: int):
        self._num_steps = num_steps
        self._step = 0

        self.observations = torch.zeros(num_steps + 1, num_envs, *observation_shape)
        self.actions = torch.zeros(num_steps, num_envs, 1, dtype=torch.long)
        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_predictions = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps, num_envs, 1)
        self.masks = torch.ones(num_steps + 1, num_envs, 1)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1,
                                                   num_envs,
                                                   recurrent_hidden_state_size)

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
        prev_actions = self.actions[step - 1]
        prev_rewards = self.rewards[step - 1]
        return states, rnn_hxs, masks, prev_actions, prev_rewards

    def get_critic_input(self):
        return self.get_actor_input(step=-1)
