import os
import json
import h5py
import numpy as np
from tqdm import tqdm

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

    for robot, obj, obst, task in tqdm(
        zip(robots, objs, obsts, tasks), desc="Loading Data"
    ):
        dataset = load_single_dataset(base_path, dataset_type, robot, obj, obst, task)
        for k in KEYS:
            data_dict[k].append(dataset[k])
            break
        break

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


def get_task_list(base_path, dataset_type, split, holdout_elem, seed):
    if split == "holdout":
        exp_name = f"{dataset_type}_{holdout_elem}_{split}_{seed}"
        task_list_path = os.path.join(
            base_path, f"{split}/{holdout_elem}_split_{seed}.json"
        )
    else:
        exp_name = f"{dataset_type}_{split}_{seed}"
        task_list_path = os.path.join(base_path, f"{split}/split_{seed}.json")

    with open(task_list_path, "r") as f:
        task_list = json.load(f)

    train_task_list = task_list["train"]
    test_task_list = task_list["test"]
    if split == "compositional":
        expert_task_list = task_list["expert"]
        train_task_list = [x for x in train_task_list if x not in expert_task_list]
    else:
        expert_task_list = None

    return exp_name, train_task_list, expert_task_list, test_task_list


def get_partial_task_list(task_list, remove_elems, n_tasks=-1):
    partial_task_list = []
    other_tasks = []
    for task in task_list:
        robot, obj, obst, subtask = task
        if (
            robot in remove_elems
            or obj in remove_elems
            or obst in remove_elems
            or subtask in remove_elems
        ):
            other_tasks.append(task)
            continue
        partial_task_list.append(task)

    assert (
        len(partial_task_list) >= n_tasks
    ), "Not enough tasks to return from partial list. Need to lower count of remove_elems or n_tasks."
    partial_task_list = np.array(partial_task_list)

    if n_tasks == -1:
        return partial_task_list, other_tasks
    return partial_task_list[:n_tasks], other_tasks
