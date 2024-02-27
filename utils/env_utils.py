from tqdm import tqdm
import numpy as np

from composuite.env.gym_wrapper import GymWrapper

from typing import Sequence
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

# fmt: off
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# fmt: on

class CompoSuiteGymnasiumWrapper(GymWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.step_counter = 0

    def reset(self):
        obs = super().reset()
        self.step_counter = 0
        return obs, {}

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.step_counter == self.horizon - 1:
            truncated = True
            self.step_counter = 0
        else:
            truncated = False
            self.step_counter += 1

        return obs, reward, done, truncated, info


class VecEnv(ABC):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self, seed=None):
        pass

    @abstractmethod
    def step_async(self, actions):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def get_attr(self, attr_name, indices=None):
        pass

    @abstractmethod
    def set_attr(self, attr_name, value, indices=None):
        pass

    @abstractmethod
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def get_images(self) -> Sequence[np.ndarray]:
        raise NotImplementedError

    def render(self, mode: str = "human"):
        raise NotImplementedError

    def getattr_depth_check(self, name, already_found):
        if hasattr(self, name) and already_found:
            return "{0}.{1}".format(type(self).__module__, type(self).__name__)
        else:
            return None

    def _get_indices(self, indices):
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in tqdm(env_fns)]
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        super().__init__(
            len(self.envs),
            self.envs[0].observation_space,
            self.envs[0].action_space,
        )

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        return_list = [env.step(a) for (env, a) in zip(self.envs, self.actions)]
        obs, rews, dones, _, infos = map(np.array, zip(*return_list))
        return obs, rews, dones, infos

    def seed(self, seed=None):
        raise NotImplementedError

    def reset(self, seed=None):
        obs_list = [env.reset()[0] for env in self.envs]
        return np.array(obs_list)

    def close(self):
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[np.ndarray]:
        return [env.render(mode="rgb_array") for env in self.envs]

    def render(self, mode: str = "human"):
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [
            getattr(env_i, method_name)(*method_args, **method_kwargs)
            for env_i in target_envs
        ]

    def _get_target_envs(self, indices):
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
