import multiprocessing as mp
from collections import deque
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


class BufferFrameEnvWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, n_frames: int = 4):
        super().__init__(env)
        _, height, width = env.observation_space.shape
        observation_shape = (n_frames, height, width)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=observation_shape,
                                                dtype=np.uint8)
        self._buffer = deque(maxlen=n_frames)
        self._n_frames = n_frames

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._buffer.append(observation)
        total_reward = reward
        for _ in range(self._n_frames - 1):
            if not done:
                observation, reward, done, info = self.env.step(action)
                total_reward += reward
            self._buffer.append(observation)
        return self._get_stacked_observations(), total_reward, done, info

    def reset(self):
        self._buffer.clear()
        observation = self.env.reset()
        for _ in range(self._n_frames):
            self._buffer.append(observation)
        return self._get_stacked_observations()

    def _get_stacked_observations(self):
        return np.stack(self._buffer, axis=0).reshape(self.observation_space.shape)


class ReshapeRewardEnvWrapper(gym.Wrapper):

    def __init__(self, env, score_reward_weigh: float = 0.025):
        super().__init__(env)
        self._prev_score = 0
        self._score_reward_weight = score_reward_weigh

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        # Include in-game score into reward.
        reward += (info['score'] - self._prev_score) * self._score_reward_weight
        self._prev_score = info['score']

        if done:
            reward += 50 if info['flag_get'] else -50

        reward = np.clip(reward, -15, 15)
        return observation, reward, done, info

    def reset(self):
        return self.env.reset()


def create_environment(env_name: str = 'SuperMarioBros-v0') -> gym.Env:
    env = gym_super_mario_bros.make(env_name)
    env = ResizeFrameEnvWrapper(env, grayscale=True)
    env = BufferFrameEnvWrapper(env, n_frames=4)
    env = ReshapeRewardEnvWrapper(env)
    env = BinarySpaceToDiscreteSpaceEnv(env, actions.COMPLEX_MOVEMENT)
    return env


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
                # Scale reward in range (-1, 1). 15 is the maximum reward.
                remote.send((observation, reward / 15.0, done, info))
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

    def __init__(self, num_envs: int):
        self._closed = False
        self._remotes, self._work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self._processes = []

        for work_remote, remote in zip(self._work_remotes, self._remotes):
            env = create_environment()
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
