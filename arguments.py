import argparse


N_JOBS = 32
STEPS_PER_UPDATE = 128
TOTAL_STEPS = STEPS_PER_UPDATE * N_JOBS * 2500
SAVE_INTERVAL = 250
HIDDEN_LAYER_SIZE = 512
RECURRENT_HIDDEN_LAYER_SIZE = 512
PREV_ACTIONS_HIDDEN_LAYER_SIZE = 256
DISCOUNT_FACTOR = 0.99
GAE_LAMBDA = 0.95
LR = 5.5e-4
MAX_GRAD_NORM = 0.5
POLICY_LOSS_COEF = 1.0
VALUE_LOSS_COEF = 0.5
ENTROPY_LOSS_COEF = 1e-3
PPO_CLIP_THRESHOLD = 0.2
PPO_EPOCHS = 4
PPO_MINIBATCHES = 16


def parse_args():
    parser = argparse.ArgumentParser('Super Mario RL Agent')
    parser.add_argument(
        '-w', '--world',
        default=1, type=int,
        help='Super Mario Bros world.'
    )
    parser.add_argument(
        '-s', '--stage',
        default=1, type=int,
        help='Super Mario Bros stage.'
    )
    parser.add_argument(
        '-j', '--jobs',
        default=N_JOBS, type=int,
        help='Number of parallel environment jobs/processes.'
    )
    parser.add_argument(
        '--steps-per-update',
        default=STEPS_PER_UPDATE, type=int,
        help='Number of steps before agent network weights are updated.'
    )
    parser.add_argument(
        '--steps',
        default=TOTAL_STEPS, type=int,
        help='Number of total environment steps.'
    )
    parser.add_argument(
        '--save-interval',
        default=SAVE_INTERVAL, type=int,
        help='Number of steps before agent network weights are saved.'
    )
    parser.add_argument(
        '--hidden-size',
        default=HIDDEN_LAYER_SIZE, type=int,
        help='Size of agent network hidden layer.'
    )
    parser.add_argument(
        '--recurrent-hidden-size',
        default=RECURRENT_HIDDEN_LAYER_SIZE,
        type=int, help='Size of agent network recurrent hidden layer.'
    )
    parser.add_argument(
        '--prev-actions-hidden-size',
        default=PREV_ACTIONS_HIDDEN_LAYER_SIZE, type=int,
        help='Size of previous taken actions hidden layer.'
    )
    parser.add_argument(
        '--discount',
        default=DISCOUNT_FACTOR, type=float,
        help='Reward decay gamma.'
    )
    parser.add_argument(
        '--gae-lambda',
        default=GAE_LAMBDA, type=float,
        help='GAE lambda parameter.'
    )
    parser.add_argument(
        '--lr',
        default=LR, type=float,
        help='Learning rate.'
    )
    parser.add_argument(
        '--max-grad-norm',
        default=MAX_GRAD_NORM, type=float,
        help='Maximum gradient norm.'
    )
    parser.add_argument(
        '--policy-loss-coef',
        default=POLICY_LOSS_COEF, type=float,
        help='Policy loss weight coefficient.'
    )
    parser.add_argument(
        '--value-loss-coef',
        default=VALUE_LOSS_COEF, type=float,
        help='Value loss weight coefficient.'
    )
    parser.add_argument(
        '--entropy-loss-coef',
        default=ENTROPY_LOSS_COEF, type=float,
        help='Entropy loss weight coefficient.'
    )
    parser.add_argument(
        '--ppo-clip-threshold',
        default=PPO_CLIP_THRESHOLD, type=float,
        help='PPO policy loss clipping threshold.'
    )
    parser.add_argument(
        '--ppo-epochs',
        default=PPO_EPOCHS, type=int,
        help='Number of epochs in PPO algorithm during update.'
    )
    parser.add_argument(
        '--ppo-minibatches',
        default=PPO_MINIBATCHES, type=int,
        help='Number of minibatches in PPO algorithm during update.'
    )
    return parser.parse_args()
