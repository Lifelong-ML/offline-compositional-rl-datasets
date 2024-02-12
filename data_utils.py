import os
import h5py
import numpy as np

from d3rlpy.datasets import MDPDataset

AVAILABLE_ROBOTS = ["IIWA", "Jaco", "Kinova3", "Panda"]
AVAILABLE_OBSTACLES = ["None", "GoalWall", "ObjectDoor", "ObjectWall"]
AVAILABLE_OBJECTS = ["Box", "Dumbbell", "Plate", "Hollowbox"]
AVAILABLE_TASKS = ["PickPlace", "Push", "Shelf", "Trashcan"]

KEYS = ["observations", "actions", "rewards", "terminals", "infos"]


def assert_config_valid(robot, obj, obst, task):
    assert robot in AVAILABLE_ROBOTS, "Robot not available"
    assert obj in AVAILABLE_OBJECTS, "Object not available"
    assert obst in AVAILABLE_OBSTACLES, "Obstacle not available"
    assert task in AVAILABLE_TASKS, "Task not available"


def load_single_dataset(base_path, dataset_type, robot, obj, obst, task):
    assert_config_valid(robot, obj, obst, task)
    data_path = os.path.join(
        base_path, dataset_type, f"{robot}_{obj}_{obst}_{task}.hdf5", "data.hdf5"
    )

    data_dict = {}

    shapes = [93, 8, 1, 1, 1]
    with h5py.File(data_path, "r") as dataset_file:
        for k, shape in zip(KEYS, shapes):
            assert k in dataset_file, f"Key {k} not found in dataset"
            data_dict[k] = dataset_file[k][:]
            assert len(data_dict[k]) == 1000000, f"Key {k} has wrong length"
            assert data_dict[k].shape[1] == shape, f"Key {k} has wrong shape"

    return data_dict


def load_multiple_datasets(base_path, dataset_type, robots, objs, obsts, tasks):
    assert (
        len(robots) == len(objs) == len(obsts) == len(tasks)
    ), "All lists must have the same length"
    for robot, obj, obst, task in zip(robots, objs, obsts, tasks):
        assert_config_valid(robot, obj, obst, task)

    data_dict = {}
    for k in KEYS:
        data_dict[k] = []

    for robot, obj, obst, task in zip(robots, objs, obsts, tasks):
        dataset = load_single_dataset(base_path, dataset_type, robot, obj, obst, task)
        for k in KEYS:
            data_dict[k].append(dataset[k])

    return data_dict


def dataset_to_mdpdataset(dataset):
    for key in KEYS:
        if len(dataset[key].shape) == 3:
            dataset[key] = np.concatenate(dataset[key], axis=0)

    episode_terminals = np.logical_or(dataset["terminals"], dataset["timeouts"])
    mdp_dataset = MDPDataset(
        observations=dataset["observations"],
        actions=dataset["actions"],
        rewards=dataset["rewards"],
        terminals=dataset["terminals"],
        episode_terminals=episode_terminals,
    )

    return mdp_dataset
