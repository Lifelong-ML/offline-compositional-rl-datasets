from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast

import os
import json
import numpy as np
import gym

from d3rlpy.metrics.scorer import AlgoProtocol
import torch

import d3rlpy
from d3rlpy.torch_utility import set_state_dict, to_cuda

from offline_env_with_envs import MTLOfflineCompoSuiteEnv
from compositional_encoder import CompositionalIQL, CompositionalEncoderFactory
from offline_env import MTLOfflineCompoSuiteEnv


def load_model(path, number: int = 300000, algo: str = "iql"):
    """Load a default model from a given path.

    Args:
        path (str): Path to the model.
        number (int, optional): Number of the model to load. Defaults to 300000.
        algo (str, optional): Algorithm to load. Defaults to "iql".

    Returns:
        d3rlpy.algos.AlgoBase: The loaded model.
    """
    if algo == "iql":
        trainer = d3rlpy.algos.IQL.from_json(os.path.join(path, "params.json"))
    elif algo == "bc":
        trainer = d3rlpy.algos.BC.from_json(os.path.join(path, "params.json"))
    else:
        raise NotImplementedError

    trainer.load_model(os.path.join(path, f"model_{number}.pt"))
    to_cuda(trainer._impl, "cuda")

    return trainer


def load_compositional_model(path, number: int = 300000):
    """Load a compositional model from a given path.

    Args:
        path (str): Path to the model.
        number (int, optional): Number of the model to load. Defaults to 300000.

    Returns:
        CompositionalIQL: The loaded model.
    """
    obs_dim = 93
    act_dim = 8
    observation_positions = {
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

    sizes = ((32,), (32, 32), (64, 64, 64), (64, 64, 64))
    module_names = ["obstacle_id", "object_id", "subtask_id", "robot_id"]
    module_input_names = [
        "obstacle-state",
        "object-state",
        "goal-state",
        "robot0_proprio-state",
    ]
    module_assignment_positions = [observation_positions[key] for key in module_names]
    interface_depths = [-1, 1, 2, 1]
    graph_structure = [[0], [1, 2], [3]]
    num_modules = [len(onehot_pos) for onehot_pos in module_assignment_positions]
    module_inputs = [observation_positions[key] for key in module_input_names]

    train_kwargs = {
        "trainer_kwargs": {
            "batch_size": int(256),
        },
        "fit_kwargs": {"save_interval": 10},
    }

    encoder_kwargs = {
        "sizes": sizes,
        "obs_dim": obs_dim,
        "action_dim": act_dim,
        "num_modules": num_modules,
        "module_assignment_positions": module_assignment_positions,
        "module_inputs": module_inputs,
        "interface_depths": interface_depths,
        "graph_structure": graph_structure,
    }
    actor_encoder_factory = CompositionalEncoderFactory(encoder_kwargs=encoder_kwargs)
    critic_encoder_factory = CompositionalEncoderFactory(encoder_kwargs=encoder_kwargs)
    trainer = CompositionalIQL(
        **train_kwargs["trainer_kwargs"],
        use_gpu=True,
        actor_encoder_factory=actor_encoder_factory,
        critic_encoder_factory=critic_encoder_factory,
    )

    trainer._create_impl((obs_dim,), act_dim)
    helper = torch.load(os.path.join(path, f"model_{number}.pt"), map_location="cpu")

    set_state_dict(trainer._impl, helper)
    to_cuda(trainer._impl, "cuda")

    return trainer


def load_tasklist(dataset_split: str, data_seed: int, task_list_path: str):
    """Given a dataset split and a data seed, load the corresponding task list.

    Args:
        dataset_split (str): Dataset split, either "train" or "test".
        data_seed (int): Data seed.
        task_list_path (str): Path to the task list.

    Returns:
        train_task_list (list): List of training tasks.
        test_task_list (list): List of testing tasks.
    """
    task_list_path = os.path.join(
        task_list_path, f"{dataset_split}/split_{data_seed}.json"
    )

    with open(task_list_path, "r") as f:
        json_file = json.load(f)
        train_task_list = json_file["train"]
        test_task_list = json_file["test"]

    return train_task_list, test_task_list


def custom_evaluate_on_environment(
    env: gym.Env,
    n_trials: int = 10,
    epsilon: float = 0.0,
    render: bool = False,
    num_steps=500,
) -> Callable[..., float]:
    observation_shape = env.observation_space.shape

    def scorer(algo: AlgoProtocol, *args: Any) -> float:
        episode_rewards = []

        for n in range(n_trials):
            observation = env.reset()

            episode_reward = np.zeros(observation.shape[0])
            for i in range(num_steps):
                # take action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = algo.predict(observation)

                observation, reward, done, _ = env.step(action)
                episode_reward += reward

                done_indices = np.where(done)[0]
                if done_indices is not None:
                    env.VecEnvList[0].env_method("reset", indices=done_indices)

            episode_rewards.append(episode_reward)
        return float(np.mean(episode_rewards))

    return scorer


def test():
    task_list = [["IIWA", "Box", "None", "Push"], ["IIWA", "Box", "None", "Push"]]
    offline_kwargs = {"ref_min_score": 0, "ref_max_score": 500, "dataset_url": ""}

    vec_env = MTLOfflineCompoSuiteEnv(
        task_list=task_list, **({"offline_kwargs": offline_kwargs})
    )

    class RandomPredictor:
        def __init__(self):
            pass

        def predict(self, obs):
            action = np.random.random(size=(1, len(task_list), 8))
            return action, None

    rp = RandomPredictor()
    reward = custom_evaluate_on_environment(vec_env)(rp)
    print(reward)


if __name__ == "__main__":
    test()
