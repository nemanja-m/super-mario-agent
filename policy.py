import torch
import torch.nn as nn
from torch.distributions import Categorical


def _init_module_weights(module: nn.Module, gain='relu') -> nn.Module:
    gain_init = 1 if gain == 'constant' else nn.init.calculate_gain(gain)
    nn.init.orthogonal_(module.weight.data, gain=gain_init)
    nn.init.constant_(module.bias.data, 0)
    return module


def _init_gru(gru_module):
    for name, param in gru_module.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param)
    return gru_module


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BasePolicy(nn.Module):
    def __init__(self,
                 state_frame_channels: int,
                 action_space_size: int,
                 hidden_layer_size: int,
                 prev_actions_out_size: int,
                 device: torch.device):
        super().__init__()

        self._hidden_layer_size = hidden_layer_size
        self._action_space_size = action_space_size
        self._prev_actions_out_size = prev_actions_out_size
        self._device = device
        self.is_recurrent = False

        self._cnn = nn.Sequential(
            _init_module_weights(nn.Conv2d(state_frame_channels, 32, 8, stride=4)),
            nn.ReLU(),
            _init_module_weights(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            _init_module_weights(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU())

        self._flatten = Flatten()

        # Last 4 actions are stacked.
        self._prev_action_linear = nn.Sequential(
            _init_module_weights(nn.Linear(4 * action_space_size,
                                           prev_actions_out_size)),
            nn.ReLU()
        )

        self._linear = nn.Sequential(
            _init_module_weights(nn.Linear(32 * 7 * 8 + prev_actions_out_size,
                                           hidden_layer_size)),
            nn.ReLU()
        )

        self.train()

    def forward(self, input_states, masks, prev_actions, rnn_hxs):
        cnn_out = self._cnn(input_states.type(torch.float32) / 255.0)
        flat_out = self._flatten(cnn_out)
        prev_actions_out = self._prev_action_linear(prev_actions)
        linear_in = torch.cat((flat_out, prev_actions_out), dim=1)
        linear_out = self._linear(linear_in)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(linear_out, rnn_hxs, masks)
        else:
            x = linear_out

        return self._critic_linear(x), x, rnn_hxs

    def act(self, input_states, rnn_hxs, masks, prev_actions):
        value, actor_features, rnn_hxs = self._base_forward(input_states,
                                                            masks,
                                                            prev_actions,
                                                            rnn_hxs)
        action, action_log_prob, action_entropy = self._sample_action(actor_features)
        return value, action, action_log_prob, action_entropy, rnn_hxs

    def value(self, input_states, rnn_hxs, masks, prev_actions):
        value, _, _ = self._base_forward(input_states,
                                         masks,
                                         prev_actions,
                                         rnn_hxs)
        return value.detach()

    def evaluate_actions(self,
                         input_states,
                         rnn_hxs,
                         masks,
                         prev_actions,
                         actions):
        value, actor_features, rnn_hxs = self._base_forward(input_states,
                                                            masks,
                                                            prev_actions,
                                                            rnn_hxs)
        distribution = self._action_distribution(actor_features)
        action_log_probs = distribution.log_prob(actions)
        action_entropy = distribution.entropy().mean()
        return value, action_log_probs, action_entropy

    def _base_forward(self, input_states, masks, prev_actions, rnn_hxs):
        prev_actions_tensor = self._create_prev_actions_tensor(prev_actions)
        value, actor_features, rnn_hxs = self(input_states,
                                              masks,
                                              prev_actions_tensor,
                                              rnn_hxs)
        return value, actor_features, rnn_hxs

    def _sample_action(self, actor_features):
        distribution = self._action_distribution(actor_features)
        action = distribution.sample()
        action_log_prob = distribution.log_prob(action)
        action_entropy = distribution.entropy().mean()
        return (action.unsqueeze(-1).long(),
                action_log_prob.unsqueeze(-1),
                action_entropy)

    def _action_distribution(self, actor_features):
        actor_out = self._actor_linear(actor_features)
        action_logits = nn.functional.log_softmax(actor_out, dim=1)
        distribution = Categorical(logits=action_logits)
        return distribution

    def _create_prev_actions_tensor(self, prev_actions):
        batch_size, stack_size = prev_actions.size(0), prev_actions.size(1)
        prev_actions_tensor = torch.zeros(
            batch_size,
            stack_size,
            self._action_space_size
        ).to(self._device)
        prev_actions_tensor.scatter_(2, prev_actions, 1)  # one-hot encoded actions
        return prev_actions_tensor.view(batch_size, -1)


class NonRecurrentPolicy(BasePolicy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._critic_linear = _init_module_weights(
            nn.Linear(self._hidden_layer_size, 1),
            gain='constant'
        )
        self._actor_linear = _init_module_weights(
            nn.Linear(self._hidden_layer_size, self._action_space_size),
            gain='constant'
        )
        self.to(self._device)


class RecurrentPolicy(BasePolicy):

    def __init__(self, *args, **kwargs):
        self._recurrent_hidden_size = kwargs.pop('recurrent_hidden_size')
        super().__init__(*args, **kwargs)

        self._gru = _init_gru(nn.GRU(input_size=self._hidden_layer_size,
                                     hidden_size=self._recurrent_hidden_size))
        self._critic_linear = _init_module_weights(
            nn.Linear(self._recurrent_hidden_size, 1),
            gain='constant'
        )
        self._actor_linear = _init_module_weights(
            nn.Linear(self._recurrent_hidden_size, self._action_space_size),
            gain='constant'
        )
        self.is_recurrent = True
        self.to(self._device)

    def _forward_gru(self, x, rnn_hxs, masks):
        if x.size(0) == rnn_hxs.size(0):
            x, hxs = self._gru(x.unsqueeze(0), (rnn_hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            num_envs_per_batch = rnn_hxs.size(0)
            steps_per_update = int(x.size(0) / num_envs_per_batch)

            x = x.view(steps_per_update, num_envs_per_batch, x.size(1))
            masks = masks.view(steps_per_update, num_envs_per_batch)

            # Figure out which steps in the sequence have a zero for any agent
            # Always assume t=0 has a zero in it as that makes the logic
            # cleaner.
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # Add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [steps_per_update]

            hxs = rnn_hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # Process steps that don't have any zeros in masks together.
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self._gru(x[start_idx:end_idx],
                                            hxs * masks[start_idx].view(1, -1, 1))
                outputs.append(rnn_scores)

            x = torch.cat(outputs, dim=0).view(steps_per_update * num_envs_per_batch, -1)
            hxs = hxs.squeeze(0)

        return x, hxs
