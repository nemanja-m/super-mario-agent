import multiprocessing as mp
from collections import deque, OrderedDict

import numpy as np
import torch
from tqdm import tqdm

from environment import MultiprocessEnvironment
from experience import ExperienceStorage
from policy import RecurrentPolicy
from ppo import PPOAgent


MAX_X = 3161


def learn(num_envs: int,
          device: torch.device,  # CUDA or CPU
          total_steps: int = 512 * 8 * 2048,
          steps_per_update: int = 512,
          hidden_layer_size: int = 512,
          recurrent_hidden_size: int = 512,
          discount=0.98,
          gae_lambda=0.95,
          save_interval=256):
    envs = MultiprocessEnvironment(num_envs=num_envs)
    actor_critic = RecurrentPolicy(state_frame_channels=envs.observation_shape[0],
                                   action_space_size=envs.action_space_size,
                                   hidden_layer_size=hidden_layer_size,
                                   prev_actions_out_size=64,
                                   recurrent_hidden_size=recurrent_hidden_size,
                                   device=device)
    experience_storage = ExperienceStorage(num_steps=steps_per_update,
                                           num_envs=num_envs,
                                           observation_shape=envs.observation_shape,
                                           recurrent_hidden_size=recurrent_hidden_size,
                                           device=device)
    agent = PPOAgent(actor_critic)

    initial_observations = envs.reset()
    experience_storage.insert_initial_observations(initial_observations)

    num_updates = total_steps // (num_envs * steps_per_update)
    episode_rewards = deque(maxlen=16)

    for update_step in tqdm(range(num_updates)):
        for step in range(steps_per_update):
            with torch.no_grad():
                actor_input = experience_storage.get_actor_input(step)
                (values,
                 actions,
                 action_log_probs,
                 _,  # Action disribution entropy is not needed.
                 recurrent_hidden_states) = actor_critic.act(*actor_input)

                observations, rewards, done_values, info_dicts = envs.step(actions)
                masks = 1 - done_values
                experience_storage.insert(observations,
                                          actions,
                                          action_log_probs,
                                          rewards,
                                          values,
                                          masks,
                                          recurrent_hidden_states)

            for done, info in zip(done_values, info_dicts):
                if done:
                    level_completed_percentage = info['x_pos'] / MAX_X
                    episode_rewards.append(level_completed_percentage)

        with torch.no_grad():
            critic_input = experience_storage.get_critic_input()
            next_value = actor_critic.value(*critic_input)

        experience_storage.compute_returns(next_value,
                                           discount=discount,
                                           gae_lambda=gae_lambda)

        losses = agent.update(experience_storage)

        if episode_rewards:
            with torch.no_grad():
                cumulative_reward = experience_storage.rewards.sum((0, 2))
                mean_reward = cumulative_reward.mean().item()
                std_reward = cumulative_reward.std().item()

            print('\n')
            metrics = OrderedDict(
                mean_x=np.mean(episode_rewards),
                median_x=np.median(episode_rewards),
                max_x=np.max(episode_rewards),
                min_x=np.min(episode_rewards),
                mean_reward=mean_reward,
                std_reward=std_reward,
                policy_loss=losses['policy_loss'],
                value_loss=losses['value_loss'],
                action_dist_entropy=losses['action_dist_entropy']
            )
            for metric, value in metrics.items():
                print('{}: {:.3f}'.format(metric, value))
            print()

        save_model = (update_step % save_interval) == (save_interval - 1)
        if save_model:
            model_path = 'models/model_{}.bin'.format(update_step + 1)
            torch.save(actor_critic.state_dict(), model_path)


if __name__ == '__main__':
    cpu_count = mp.cpu_count()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    learn(num_envs=cpu_count, device=device)
