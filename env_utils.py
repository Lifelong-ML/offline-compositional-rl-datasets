import os
import json
import numpy as np

from composuite.env.gym_wrapper import GymWrapper


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
