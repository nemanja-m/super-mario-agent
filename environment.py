import multiprocessing as mp
from multiprocessing.connection import Connection

import cv2
import gym
import gym_super_mario_bros
import numpy as np
import torch
from gym_super_mario_bros import actions
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv


class ResizeFrameEnvWrapper(gym.ObservationWrapper):
    def __init__(self,
                 env: gym.Env,
                 width: int = 96,
                 height: int = 90,
                 grayscale: bool = False):
        """Resize env frames to width x height.

        State image dimensions are transposed to match pytorch image dimension
        conventions.  Pytorch uses (channels, height, width) image shape.

        :param env: (Gym Environment) the environment

        """
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        channels = 1 if grayscale else 3
        shape = (channels, self.height, self.width)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=shape,
                                                dtype=env.observation_space.dtype)

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """Returns the downsampled, current observation from a frame."""
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)

        if self.grayscale:
            frame = frame[:, :, None]

        # pytorch uses channel first images
        return frame.transpose(2, 0, 1)


class StochasticFrameSkipEnvWrapper(gym.Wrapper):

    def __init__(self, env, n_frames: int = 4, action_stick_prob: float = 0.25):
        super().__init__(env)
        self._n_frames = n_frames
        self._action_stick_prob = action_stick_prob
        self._current_action = None

    def reset(self):
        self._current_action = None
        return self.env.reset()

    def step(self, action):
        done = False
        total_reward = 0
        for frame in range(self._n_frames):
            if self._current_action is None:
                self._current_action = action
            elif frame == 0:
                if np.random.rand() > self._action_stick_prob:
                    self._current_action = action
            elif frame == 1:
                self._current_action = action
            observation, reward, done, info = self.env.step(self._current_action)
            total_reward += reward
            if done:
                break
        return observation, total_reward, done, info


class ReshapeRewardEnvWrapper(gym.Wrapper):

    def __init__(self,
                 env: gym.Env,
                 max_delta_x: int = 2,
                 score_reward_weight: float = 0.025,
                 time_reward_weight: float = -0.9):
        super().__init__(env)
        self._max_delta_x = max_delta_x
        self._score_reward_weight = score_reward_weight
        self._time_reward_weight = time_reward_weight
        self._prev_score = 0
        self._prev_x = 40  # Starting Mario position.
        self._prev_time = 400

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        delta_x = info['x_pos'] - self._prev_x - 0.05
        reward = np.clip(delta_x, -self._max_delta_x, self._max_delta_x)
        self._prev_x = info['x_pos']

        reward += (self._prev_time - info['time']) * self._time_reward_weight
        self._prev_time = info['time']

        # Include in-game score into reward.
        reward += (info['score'] - self._prev_score) * self._score_reward_weight
        self._prev_score = info['score']

        if done:
            reward += 500 if info['flag_get'] else -50

        reward /= 10.0
        return observation, reward, done, info

    def reset(self):
        return self.env.reset()


def build_environment(mario_env_name: str, action_space):
    env = gym_super_mario_bros.make(mario_env_name)
    env = ResizeFrameEnvWrapper(env, grayscale=True)
    env = ReshapeRewardEnvWrapper(env)
    env = StochasticFrameSkipEnvWrapper(env, n_frames=4)
    return BinarySpaceToDiscreteSpaceEnv(env, action_space)


def _worker(remote: Connection,
            parent_remote: Connection,
            env: gym.Env) -> None:
    parent_remote.close()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'render':
                env.render()
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError
        except EOFError:
            break


class MultiprocessEnvironment:

    def __init__(self,
                 num_envs: int,
                 mario_env_name: str = 'SuperMarioBros-1-1-v0',
                 action_space=actions.COMPLEX_MOVEMENT):

        self._closed = False
        self._remotes, self._work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self._processes = []

        for work_remote, remote in zip(self._work_remotes, self._remotes):
            env = build_environment(mario_env_name, action_space)
            args = (work_remote, remote, env)
            process = mp.Process(target=_worker, args=args, daemon=True)
            process.start()
            self._processes.append(process)
            work_remote.close()

        self.action_space_size = env.action_space.n
        self.observation_shape = env.observation_space.shape

    def step(self, actions: torch.Tensor):
        for remote, action in zip(self._remotes, actions):
            remote.send(('step', action.item()))

        observations, rewards, dones, infos = zip(*[remote.recv()
                                                    for remote in self._remotes])
        return (torch.from_numpy(np.stack(observations)),
                torch.from_numpy(np.stack(rewards)).unsqueeze(-1),
                torch.from_numpy(np.stack(dones).astype(np.uint8)).unsqueeze(-1),
                np.stack(infos))

    def reset(self):
        for remote in self._remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self._remotes]
        return torch.from_numpy(np.stack(obs))

    def render(self) -> None:
        for remote in self._remotes:
            remote.send(('render', None))

    def close(self) -> None:
        if self._closed:
            return

        for remote in self._remotes:
            remote.send(('close', None))

        for process in self._processes:
            process.join()

        self._closed = True
