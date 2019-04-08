import argparse
import os

import torch

from environment import create_environment
from policy import RecurrentPolicy


HIDDEN_LAYER_SIZE = 512
PREV_ACTIONS_HIDDEN_SIZE = 128
RECURRENT_HIDDEN_SIZE = 64

_root_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_root_dir, 'models', 'latest.bin')


def _env_name(level: str) -> str:
    env_name = '{env}-{level}-{version}'.format(env='SuperMarioBros',
                                                level=level,
                                                version='v0')
    return env_name


def run(hidden_layer_size: int, model_path: str, level: str):
    env = create_environment(env_name=_env_name(level))
    actor_critic = RecurrentPolicy(state_frame_channels=env.observation_space.shape[0],
                                   action_space_size=env.action_space.n,
                                   hidden_layer_size=hidden_layer_size,
                                   prev_actions_out_size=PREV_ACTIONS_HIDDEN_SIZE,
                                   recurrent_hidden_size=RECURRENT_HIDDEN_SIZE,
                                   device=torch.device('cpu'))
    actor_critic.load_state_dict(torch.load(model_path, map_location='cpu'))

    observation = env.reset()
    rnn_hxs = torch.zeros(1, RECURRENT_HIDDEN_SIZE)
    masks = torch.zeros(1, 1)
    prev_actions = torch.zeros(1, 4, 1, dtype=torch.long)

    try:
        while True:
            with torch.no_grad():
                observation = torch.from_numpy(observation).unsqueeze(0)
                value, action, _, _, rnn_hxs = actor_critic.act(observation,
                                                                rnn_hxs,
                                                                masks,
                                                                prev_actions)
            observation, reward, done, _ = env.step(action.flatten().item())
            masks = 1 - torch.tensor(done, dtype=torch.uint8)
            prev_actions = torch.roll(prev_actions, 1, dims=1)
            prev_actions[0, -1, 0] = action.item()

            if done:
                observation = env.reset()

            env.render()
    except (KeyboardInterrupt, EOFError):
        print('\n\nExiting Super Mario Bros. GLHF.')


def parse_args():
    parser = argparse.ArgumentParser('Run trained Super Mario agent in environment.')
    parser.add_argument('--hidden', type=int, default=HIDDEN_LAYER_SIZE,
                        help='Size of hidden layer in reccurent policy.')
    parser.add_argument('-m', '--model', type=str, default=MODEL_PATH,
                        help='Path to trained agent model.')
    parser.add_argument('-w', '--world', type=int, default=1,
                        help='Super Mario Bros world.')
    parser.add_argument('-s', '--stage', type=int, default=1,
                        help='Super Mario Bros stage.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    assert 1 <= args.world <= 8, 'Super Mario world must be from 1 to 8.'
    assert 1 <= args.stage <= 4, 'Super Mario stage must be from 1 to 4.'

    level = '{world}-{stage}'.format(world=args.world, stage=args.stage)
    run(hidden_layer_size=args.hidden, model_path=args.model, level=level)
