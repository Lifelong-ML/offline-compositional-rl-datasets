import os
import numpy as np
import gym
from d4rl import offline_env
from d4rl.utils.wrappers import NormalizedBoxEnv
from composuite.tasks.pick_place_subtask import PickPlaceSubtask
from composuite.tasks.push_subtask import PushSubtask
from composuite.tasks.shelf_subtask import ShelfSubtask
from composuite.tasks.trashcan_subtask import TrashcanSubtask
from composuite.env.gym_wrapper import GymWrapper
from robosuite.controllers import load_controller_config
from robosuite import make
import math
import itertools
import numpy as np

from vec_env import DummyVecEnv
from gym.spaces import Box

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
    """Offline environment for CompoSuite.
    This version does not initialize the gym environment to save memory."""

    def __init__(self, **kwargs):
        """Initialize the offline environment."""
        offline_env.OfflineEnv.__init__(self, **(kwargs["offline_kwargs"]))
        self.robot = kwargs["subtask_kwargs"]["robots"]
        self.obstacle = kwargs["subtask_kwargs"]["obstacle"]
        self.object = kwargs["subtask_kwargs"]["object_type"]
        self.task = kwargs["subtask_kwargs"]["env_name"][:-7]

        self.action_space = Box(-1.0, 1.0, (8,), np.float32)
        self.observation_space = Box(-np.inf, np.inf, (93,), np.float32)
        self.observation_positions = {
            "object-state": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            "obstacle-state": np.array(
                [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
            ),
            "goal-state": np.array(
                [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
            ),
            "object_id": np.array([45, 46, 47, 48]),
            "robot_id": np.array([49, 50, 51, 52]),
            "obstacle_id": np.array([53, 54, 55, 56]),
            "subtask_id": np.array([57, 58, 59, 60]),
            # fmt: off
            "robot0_proprio-state": np.array(  
                [
                    61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                    71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                    81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                    91, 92,
                ]
            # fmt: on
            ),
        }

    def get_dataset(self, h5path=None):
        """Get the dataset from the hdf5 file.

        Args:
            h5path (str): path to the hdf5 file.

        Returns:
            dataset (h5py.Dataset): dataset from the hdf5 file.
        """
        assert h5path is not None, "Requires to pass in h5path"
        h5path = os.path.join(
            h5path,
            f"{self.robot}_{self.object}_{self.obstacle}_{self.task}",
            "data.hdf5",
        )
        return super().get_dataset(h5path)


class MTLOfflineCompoSuiteEnv(gym.Env):
    def __init__(self, max_parallel_envs=None, task_list=None, **kwargs):
        """Initialize the offline environment for multi-task learning.

        Args:
            max_parallel_envs (int): maximum number of parallel environments.
            task_list (list): list of tasks to be trained.
        """
        super().__init__()
        if task_list is None:
            task_list = all_configurations
        if max_parallel_envs == None:
            max_parallel_envs = len(task_list)

        offline_kwargs = kwargs["offline_kwargs"]
        self.max_parallel_envs = max_parallel_envs
        num_VecEnvs = math.ceil(len(task_list) / self.max_parallel_envs)
        self.VecEnvList = []

        j = 0
        for i in range(num_VecEnvs):
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
                self.VecEnvList.append(get_composuite_env(**kwargs))
                j += 1
                k += 1
        self.action_space = Box(-1.0, 1.0, (8,), np.float32)
        self.observation_space = Box(-np.inf, np.inf, (93,), np.float32)
        self.observation_positions = {
            "object-state": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            "obstacle-state": np.array(
                [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
            ),
            "goal-state": np.array(
                [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
            ),
            "object_id": np.array([45, 46, 47, 48]),
            "robot_id": np.array([49, 50, 51, 52]),
            "obstacle_id": np.array([53, 54, 55, 56]),
            "subtask_id": np.array([57, 58, 59, 60]),
            # fmt: off
            "robot0_proprio-state": np.array(  
                [
                    61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                    71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                    81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                    91, 92,
                ]
            # fmt: on
            ),
        }

    def get_normalized_score(self, score):
        raise NotImplementedError("Currently not supported.")

    @property
    def dataset_filepath(self):
        return [
            vecEnv.get_attr("dataset_filepath", indices=range(vecEnv.num_envs))
            for vecEnv in self.VecEnvList
        ]

    def get_dataset(self, h5path=None):
        """Get all the datasets for the parallel environments.

        Args:
            h5path (str): path to the hdf5 file.

        Raises:
            NotImplementedError: Requires manually setting the root path of the dataset.

        Returns:
            dataset_list (list): list of datasets for the parallel environments.
        """
        if h5path is None:
            raise NotImplementedError(
                "Requires manually setting the root path of the dataset."
            )

        else:
            dataset_list = [vecEnv.get_dataset(h5path) for vecEnv in self.VecEnvList]

        data_dict = {}

        for parallels in dataset_list:
            for key in parallels.keys():
                # for key in parallels[0].keys():
                if key not in data_dict.keys():
                    # data_dict[key] = np.r_[[dataset[key][:1000000] for dataset in parallels]]
                    data_dict[key] = np.r_[[parallels[key][:1000000]]]
                else:
                    # data_dict[key] = np.r_[[dataset[key][:1000000] for dataset in parallels], data_dict[key]]
                    data_dict[key] = np.r_[[parallels[key][:1000000]], data_dict[key]]

        return data_dict

    def get_dataset_chunk(self, chunk_id, h5path=None):
        raise NotImplementedError("Currently not supported for MTL.")

    def reset(self):
        """Reset the parallel environments."""
        return np.r_[[vecEnv.reset().squeeze() for vecEnv in self.VecEnvList]]

    def seed(self, seed=None):
        """Set the seed for the parallel environments."""
        return [vecEnv.seed(seed) for vecEnv in self.VecEnvList]

    def step(self, actions):
        """Step the parallel environments."""
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
        return observation, reward, done, information


def get_composuite_env(**kwargs):
    return OfflineCompoSuiteEnv(**kwargs)
