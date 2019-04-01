import torch.nn as nn
from torch.distributions import Categorical


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class RecurrentPolicy(nn.Module):
    def __init__(self,
                 state_frame_channels: int,
                 action_space_size: int,
                 hidden_layer_size: int = 512):
        super().__init__()

        self._cnn = nn.Sequential(
            nn.Conv2d(state_frame_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32 * 11 * 12, hidden_layer_size),
            nn.ReLU()
        )
        self._gru = nn.GRU(input_size=hidden_layer_size, hidden_size=hidden_layer_size)
        self._critic_linear = nn.Linear(hidden_layer_size, 1)
        self._actor_linear = nn.Linear(hidden_layer_size, action_space_size)

        self.train()

    def act(self, inputs, rnn_hxs, masks):
        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks)
        action, action_log_prob, action_entropy = self._sample_action(actor_features)
        return value, action, action_log_prob, action_entropy, rnn_hxs

    def forward(self, inputs, rnn_hxs, masks):
        cnn_out = self._cnn(inputs / 255.0)
        x, rnn_hxs = self._forward_gru(cnn_out, rnn_hxs, masks)
        return self._critic_linear(x), x, rnn_hxs

    def _forward_gru(self, x, rnn_hxs, masks):
        x, hxs = self._gru(x.unsqueeze(0), (rnn_hxs * masks).unsqueeze(0))
        x = x.squeeze(0)
        hxs = hxs.squeeze(0)
        return x, hxs

    def _sample_action(self, actor_features):
        actor_out = self._actor_linear(actor_features)
        action_logits = nn.functional.log_softmax(actor_out, dim=1)
        distribution = Categorical(logits=action_logits)
        action = distribution.sample()
        action_log_prob = distribution.log_prob(action)
        action_entropy = distribution.entropy().mean()
        return action, action_log_prob, action_entropy
