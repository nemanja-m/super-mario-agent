import argparse
import os
import sys
import time

import torch

from environment import build_environment
from policy import RecurrentPolicy


HIDDEN_LAYER_SIZE = 512
PREV_ACTIONS_HIDDEN_SIZE = 128
RECURRENT_HIDDEN_SIZE = 512

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def _model_path(world: int, stage: int) -> str:
    level_path = 'level_{world}_{stage}.bin'.format(world=world, stage=stage)
    return os.path.join(_ROOT_DIR, 'models', level_path)


def _env_name(world: int, stage: int) -> str:
    return '{env}-{world}-{stage}-{version}'.format(env='SuperMarioBros',
                                                    world=world,
                                                    stage=stage,
                                                    version='v0')


def run(hidden_layer_size: int, world: int, stage: int):
    model_path = _model_path(world, stage)

    if not os.path.isfile(model_path):
        err_message = '\nModel for level {world}-{stage} ' \
            'does not exist.\n'.format(world=world, stage=stage)
        sys.stderr.write(err_message)
        sys.exit(-1)

    model_weights = torch.load(model_path, map_location='cpu')

    env = build_environment(mario_env_name=_env_name(world, stage), stochastic=False)

    actor_critic = RecurrentPolicy(state_frame_channels=env.observation_space.shape[0],
                                   action_space_size=env.action_space.n,
                                   hidden_layer_size=hidden_layer_size,
                                   prev_actions_out_size=PREV_ACTIONS_HIDDEN_SIZE,
                                   recurrent_hidden_size=RECURRENT_HIDDEN_SIZE,
                                   device=torch.device('cpu'))
    actor_critic.load_state_dict(model_weights)

    observation = env.reset()
    rnn_hxs = torch.zeros(1, RECURRENT_HIDDEN_SIZE)
    masks = torch.ones(1, 1)
    prev_actions = torch.zeros(1, 4, 1, dtype=torch.long)

    try:
        while True:
            with torch.no_grad():
                observation = torch.from_numpy(observation).unsqueeze(0)
                value, action, _, _, rnn_hxs = actor_critic.act(observation,
                                                                rnn_hxs,
                                                                masks,
                                                                prev_actions)
            observation, reward, done, info = env.step(action.flatten().item())
            masks = 1 - torch.tensor(done, dtype=torch.uint8)
            prev_actions = torch.roll(prev_actions, 1, dims=1)
            prev_actions[0, -1, 0] = action.item()

            if done:
                observation = env.reset()
                if info['flag_get']:
                    print('\nExiting Super Mario Bros ...')
                    time.sleep(2)
                    raise EOFError()

            env.render()

    except (KeyboardInterrupt, EOFError):
        print('\nGLHF')


def parse_args():
    parser = argparse.ArgumentParser('Run trained Super Mario agent in environment.')
    parser.add_argument('--hidden', type=int, default=HIDDEN_LAYER_SIZE,
                        help='Size of hidden layer in recurrent policy.')
    parser.add_argument('-w', '--world', type=int, default=1,
                        help='Super Mario Bros world.')
    parser.add_argument('-s', '--stage', type=int, default=1,
                        help='Super Mario Bros stage.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    assert 1 <= args.world <= 8, 'Super Mario world must be from 1 to 8.'
    assert 1 <= args.stage <= 4, 'Super Mario stage must be from 1 to 4.'

    run(hidden_layer_size=args.hidden, world=args.world, stage=args.stage)
