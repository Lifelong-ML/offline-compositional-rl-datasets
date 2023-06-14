import math
import itertools
import os
import numpy as np
import gym

from d4rl import offline_env

from composuite.env.gym_wrapper import GymWrapper
from robosuite.controllers import load_controller_config
from robosuite import make

from vec_env import DummyVecEnv

controller_configs = load_controller_config(default_controller="JOINT_POSITION")
global_subtask_kwargs = {
    "controller_configs": controller_configs,
    "has_renderer": False,
    "has_offscreen_renderer": False,
    "reward_shaping": True,
    "use_camera_obs": False,
    "use_task_id_obs": True,
    "horizon": 500,
}

AVAILABLE_ROBOTS = ["IIWA", "Jaco", "Kinova3", "Panda"]
AVAILABLE_OBSTACLES = ["None", "GoalWall", "ObjectDoor", "ObjectWall"]
AVAILABLE_OBJECTS = ["Box", "Dumbbell", "Plate", "Hollowbox"]
AVAILABLE_TASKS = ["PickPlace", "Push", "Shelf", "Trashcan"]

all_configurations = list(
    itertools.product(
        AVAILABLE_ROBOTS, AVAILABLE_OBJECTS, AVAILABLE_OBSTACLES, AVAILABLE_TASKS
    )
)


class OfflineCompoSuiteEnv(GymWrapper, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        env = make(**global_subtask_kwargs, **(kwargs["subtask_kwargs"]))
        GymWrapper.__init__(self, env)
        offline_env.OfflineEnv.__init__(self, **(kwargs["offline_kwargs"]))
        self.robot = kwargs["subtask_kwargs"]["robots"]
        self.obstacle = kwargs["subtask_kwargs"]["obstacle"]
        self.object = kwargs["subtask_kwargs"]["object_type"]
        self.task = kwargs["subtask_kwargs"]["env_name"][:-7]

    def get_dataset(self, h5path: str = None):
        """Get the dataset from the given hdf5 path.

        Args:
            h5path (str, optional): Path to the hdf5 file. Defaults to None.

        Returns:
            d4rl.offline_env.OfflineDataset: Dataset from the given hdf5 path.
        """
        assert h5path is not None, "Requires to pass in h5path"
        h5path = os.path.join(
            h5path,
            f"{self.robot}_{self.object}_{self.obstacle}_{self.task}",
            "data.hdf5",
        )
        return super().get_dataset(h5path)


class MTLOfflineCompoSuiteEnv(gym.Env):
    def __init__(
        self,
        max_parallel_envs: int = None,
        task_list: list = None,
        step_single_env: bool = False,
        **kwargs,
    ):
        """Multi-task learning offline composuite environment. Uses a vectorized env implementation
        to run multiple environments in parallel.

        Args:
            max_parallel_envs (int, optional): Maximum number of parallel environments. Defaults to None.
            task_list (list, optional): List of tasks to train on. Defaults to None.
            step_single_env (bool, optional): Whether to step a single environment. Defaults to False.
        """
        super().__init__()
        if task_list is None:
            task_list = all_configurations
        if max_parallel_envs == None:
            max_parallel_envs = len(task_list)

        self.step_single_env = step_single_env
        self.current_env = 0
        if self.step_single_env:
            max_parallel_envs = 1

        offline_kwargs = kwargs["offline_kwargs"]
        self.max_parallel_envs = max_parallel_envs
        self.num_envs = len(task_list)
        self.num_VecEnvs = math.ceil(len(task_list) / self.max_parallel_envs)
        self.VecEnvList = []

        j = 0
        for i in range(self.num_VecEnvs):
            env_list = []
            k = 0
            while j < len(task_list) and k < max_parallel_envs:
                robot, object_type, obstacle, subtask = task_list[j]
                subtask_kwargs = {
                    "robots": robot,
                    "object_type": object_type,
                    "obstacle": obstacle,
                    "env_name": subtask + "Subtask",
                }
                kwargs = {
                    "offline_kwargs": offline_kwargs,
                    "subtask_kwargs": subtask_kwargs,
                }
                env_list.append(
                    lambda bound_kwargs=kwargs: get_composuite_env(**bound_kwargs)
                )
                j += 1
                k += 1
            self.VecEnvList.append(DummyVecEnv(env_list))
        self.action_space = self.VecEnvList[0].action_space
        self.observation_space = self.VecEnvList[0].observation_space
        self.observation_positions = self.VecEnvList[0].envs[0].observation_positions

    def get_normalized_score(self, score):
        raise NotImplementedError("Currently not supported.")

    @property
    def dataset_filepath(self):
        return [
            vecEnv.get_attr("dataset_filepath", indices=range(vecEnv.num_envs))
            for vecEnv in self.VecEnvList
        ]

    def get_dataset(self, h5path):
        """Get the dataset for all the parallel environments.

        Args:
            h5path (str): Path to the hdf5 file.

        Returns:
            dict: Dictionary of the dataset.

        Raises:
            NotImplementedError: Requires manually setting the root path of the dataset.
        """
        if h5path is None:
            raise NotImplementedError(
                "Requires manually setting the root path of the dataset."
            )

        else:
            dataset_list = [
                vecEnv.env_method("get_dataset", h5path) for vecEnv in self.VecEnvList
            ]

        data_dict = {}

        for parallels in dataset_list:
            for key in parallels[0].keys():
                if key not in data_dict.keys():
                    data_dict[key] = np.r_[[dataset[key] for dataset in parallels]]
                else:
                    data_dict[key] = np.r_[
                        [dataset[key] for dataset in parallels],
                        data_dict[key],
                    ]

        return data_dict

    def get_dataset_chunk(self, chunk_id, h5path=None):
        raise NotImplementedError("Not sure what this should be for the MTL case")

    def reset(self):
        """Reset all the parallel environments.

        Returns:
            np.ndarray: Resetted observation.
        """
        return np.r_[[vecEnv.reset().squeeze() for vecEnv in self.VecEnvList]]

    def seed(self, seed=None):
        """Seed all the parallel environments.

        Args:
            seed (int, optional): Seed value. Defaults to None.

        Returns:
            list: List of seeds.
        """
        return [vecEnv.seed(seed) for vecEnv in self.VecEnvList]

    def step(self, actions):
        """Step all the parallel environments.

        Args:
            actions (np.ndarray): Actions to take.

        Returns:
            tuple: Tuple of observations, rewards, dones, and information.
        """
        i = 0
        for vecEnv in self.VecEnvList:
            if i == 0:
                observation, reward, done, information = vecEnv.step(
                    actions[i : i + vecEnv.num_envs]
                )
            else:
                o, r, d, info = vecEnv.step(actions[i : i + vecEnv.num_envs])
                observation = np.r_[observation, o]
                reward = np.r_[reward, r]
                done = np.r_[done, d]
                information += info
            i += vecEnv.num_envs
        return observation, reward.reshape(-1, 1), done.reshape(-1, 1), information


def get_composuite_env(**kwargs):
    return OfflineCompoSuiteEnv(**kwargs)
