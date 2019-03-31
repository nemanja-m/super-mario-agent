import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Iterable

import cv2
import gym
import gym_super_mario_bros
import numpy as np
from gym_super_mario_bros import actions
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv


class ResizeFrameEnvWrapper(gym.ObservationWrapper):
    def __init__(self,
                 env: gym.Env,
                 width: int = 128,
                 height: int = 120,
                 grayscale: bool = False):
        """Warp frames to width x height.

        :param env: (Gym Environment) the environment

        """
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        channels = 1 if grayscale else 3
        shape = (channels, self.height, self.width)  # pytorch uses channel first images
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape,
                                                dtype=env.observation_space.dtype)

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """Returns the downsampled, current observation from a frame."""
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)

        if self.grayscale:
            frame = frame[:, :, None]

        return frame.transpose(2, 0, 1)  # pytorch uses channel first images


def _create_environment(env_name: str='SuperMarioBrosRandomStages-v0') -> gym.Env:
    env = gym_super_mario_bros.make(env_name)
    env = BinarySpaceToDiscreteSpaceEnv(env, actions.SIMPLE_MOVEMENT)
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

    def __init__(self, n_envs: int):
        self._closed = False
        self._remotes, self._work_remotes = zip(*[mp.Pipe() for _ in range(n_envs)])
        self._processes = []

        for work_remote, remote in zip(self._work_remotes, self._remotes):
            env = _create_environment()
            args = (work_remote, remote, env)
            process = mp.Process(target=_worker, args=args, daemon=True)
            process.start()
            self._processes.append(process)
            work_remote.close()

    def step(self, actions: Iterable[int]):
        for remote, action in zip(self._remotes, actions):
            remote.send(('step', action))

        observations, rewards, dones, infos = zip(*[remote.recv()
                                                    for remote in self._remotes])
        return observations, rewards, dones, infos

    def reset(self):
        for remote in self._remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self._remotes]
        return obs

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


if __name__ == '__main__':
    n_envs = 2
    menv = MultiprocessEnvironment(n_envs)
    states = menv.reset()
    for step in range(10000):
        res = menv.step([1] * n_envs)
        menv.render()
    menv.close()
