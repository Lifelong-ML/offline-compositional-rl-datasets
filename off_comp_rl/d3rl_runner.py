import os
import json
import numpy as np
from argparse import ArgumentParser
import torch
from compositional_encoder import CompositionalIQL, CompositionalEncoderFactory
import d3rlpy

# from composuite import make

from offline_env import MTLOfflineCompoSuiteEnv

DATASET_PATH = "_TO_SET_"
TASK_LIST_PATH = "_TO_SET_"

import os

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"


def get_dataset_and_env(
    train_task_list: list, dataset_type: str = "expert", expert_task_list: list = None
):
    """Given a list of tasks, return an MDP dataset and env for training.

    Args:
        train_task_list (list): List of tasks to train on.
        dataset_type (str, optional): Type of dataset to use. Defaults to "expert".
        expert_task_list (list, optional): List of expert tasks to use. Defaults to None.

    Returns:
        mdp_dataset (d3rlpy.dataset.MDPDataset): Dataset for training.
        env (MTLOfflineCompoSuiteEnv): Environment for training.
    """
    offline_kwargs = {"ref_min_score": 0, "ref_max_score": 500, "dataset_url": ""}
    env = MTLOfflineCompoSuiteEnv(
        task_list=train_task_list, **({"offline_kwargs": offline_kwargs})
    )
    dataset = env.get_dataset(h5path=os.path.join(DATASET_PATH, f"{dataset_type}"))

    for key in dataset.keys():
        if len(dataset[key].shape) == 2:
            dataset[key] = np.expand_dims(dataset[key], -1)
        dataset[key] = dataset[key].reshape((-1, dataset[key].shape[-1]))

    if expert_task_list:
        expert_env = MTLOfflineCompoSuiteEnv(
            task_list=expert_task_list, **({"offline_kwargs": offline_kwargs})
        )
        expert_dataset = expert_env.get_dataset(
            h5path=os.path.join(DATASET_PATH, "expert")
        )

        for key in expert_dataset.keys():
            if len(expert_dataset[key].shape) == 2:
                expert_dataset[key] = np.expand_dims(expert_dataset[key], -1)
            dataset[key] = np.concatenate(
                [
                    dataset[key],
                    expert_dataset[key].reshape((-1, expert_dataset[key].shape[-1])),
                ],
                axis=0,
            )

    episode_terminals = np.logical_or(dataset["terminals"], dataset["timeouts"])
    mdp_dataset = d3rlpy.dataset.MDPDataset(
        observations=dataset["observations"],
        actions=dataset["actions"],
        rewards=dataset["rewards"],
        terminals=dataset["terminals"],
        episode_terminals=episode_terminals,
    )

    return mdp_dataset, env


def train_algo(
    experiment_name: str,
    dataset: d3rlpy.dataset.MDPDataset,
    algo: str,
    encoder: str,
    n_steps: int,
    training_kwargs: dict,
):
    """Train an algorithm on a dataset.

    Args:
        experiment_name (str): Name of experiment.
        dataset (d3rlpy.dataset.MDPDataset): Dataset to train on.
        algo (str): Algorithm to train.
        encoder (str): Encoder to use. "compositional" or "default".
        n_steps (int): Number of steps to train for.
        training_kwargs (dict): Training kwargs.

    Raises:
        NotImplementedError: If algo is not implemented or encoder is compositional for BC.
    """

    if algo == "bc":
        if encoder == "compositional":
            raise NotImplementedError("Compositional encoder not implemented for BC.")
        else:
            trainer = d3rlpy.algos.BC(**training_kwargs["trainer_kwargs"], use_gpu=True)
    elif algo == "iql":
        if encoder == "compositional":
            encoder_kwargs = training_kwargs["encoder_kwargs"]
            actor_encoder_factory = CompositionalEncoderFactory(
                encoder_kwargs=encoder_kwargs
            )
            trainer = CompositionalIQL(
                **training_kwargs["trainer_kwargs"],
                use_gpu=True,
                actor_encoder_factory=actor_encoder_factory,
            )
        else:
            trainer = d3rlpy.algos.IQL(
                **training_kwargs["trainer_kwargs"], use_gpu=True
            )
    else:
        raise NotImplementedError("Algo not implemented.")

    trainer.fit(
        dataset,
        n_steps_per_epoch=1000,
        n_steps=n_steps,
        experiment_name=experiment_name,
        eval_episodes=dataset,
        logdir=f"d3rlpy_{encoder}_{algo}_logs",
        **training_kwargs["fit_kwargs"],
    )


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="default",
        choices=["default", "compositional", "holdout"],
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="expert",
        choices=["random", "medium", "expert", "medium-replay-subsampled"],
    )
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--use-task-list-path", default=False, action="store_true")
    parser.add_argument(
        "--holdout-elem",
        type=str,
        default="Hollowbox",
        choices=["Hollowbox", "IIWA", "ObjectWall", "PickPlace"],
    )

    parser.add_argument(
        "--algo", type=str, default="bc", choices=["bc", "bcq", "bear", "cql", "iql"]
    )

    parser.add_argument(
        "--robot",
        type=str,
        default="IIWA",
        choices=["IIWA", "Jaco", "Kinova3", "Panda"],
    )
    parser.add_argument(
        "--object",
        type=str,
        default="Box",
        choices=["Box", "HollowBox", "Plate", "Dumbbell"],
    )
    parser.add_argument(
        "--obstacle",
        type=str,
        default="None",
        choices=["None", "ObjectWall", "ObjectDoor", "GoalWall"],
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="Push",
        choices=["Push", "PickPlace", "Shelf", "Trashcan"],
    )
    parser.add_argument(
        "--encoder", type=str, default="default", choices=["compositional", "default"]
    )
    parser.add_argument("--n-steps", type=int, default=300000)

    args = parser.parse_args()
    return args


def main(
    experiment_name: str,
    train_task_list: list,
    expert_task_list: list,
    dataset_type: str,
    algo: str,
    encoder: str = "default",
    n_steps: int = 300000,
):
    """Main function to run one experiment.
    Gets dataset, sets parameters, trains algorithm.

    Args:
        experiment_name (str): Name of experiment.
        train_task_list (list): List of tasks to train on.
        expert_task_list (list): List of tasks to use expert data from.
        dataset_type (str): Type of dataset to use.
        algo (str): Algorithm to train.
        encoder (str, optional): Encoder to use. Defaults to "default".
        n_steps (int, optional): Number of steps to train for. Defaults to 300000.

    Raises:
        NotImplementedError: If algo is not implemented or encoder is compositional for BC.
    """

    num_envs = 1 if isinstance(train_task_list, dict) else len(train_task_list)
    if expert_task_list:
        num_envs += len(expert_task_list)

    train_kwargs = {
        "trainer_kwargs": {
            "batch_size": int(256 * num_envs),
        },
        "fit_kwargs": {"save_interval": 10},
    }

    dataset, env = get_dataset_and_env(
        train_task_list, dataset_type=dataset_type, expert_task_list=expert_task_list
    )

    # Random seed
    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    observation_positions = env.observation_positions

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

    train_kwargs["encoder_kwargs"] = {
        "sizes": sizes,
        "obs_dim": obs_dim,
        "action_dim": act_dim,
        "num_modules": num_modules,
        "module_assignment_positions": module_assignment_positions,
        "module_inputs": module_inputs,
        "interface_depths": interface_depths,
        "graph_structure": graph_structure,
    }

    train_algo(experiment_name, dataset, env, algo, train_kwargs, encoder, n_steps)


if __name__ == "__main__":
    args = parse_args()

    if args.use_task_list_path:
        if args.dataset_split == "holdout":
            experiment_name = f"{args.dataset_type}_{args.holdout_elem}_{args.dataset_split}_{args.data_seed}"
            task_list_path = os.path.join(
                TASK_LIST_PATH,
                f"{args.dataset_split}/{args.holdout_elem}_split_{args.data_seed}.json",
            )
        else:
            experiment_name = (
                f"{args.dataset_type}_{args.dataset_split}_{args.data_seed}"
            )
            task_list_path = os.path.join(
                TASK_LIST_PATH, f"{args.dataset_split}/split_{args.data_seed}.json"
            )

        with open(task_list_path, "r") as f:
            task_list = json.load(f)

        train_task_list = task_list["train"]
        if args.dataset_split == "compositional":
            expert_task_list = task_list["expert"]
            train_task_list = [x for x in train_task_list if x not in expert_task_list]
        else:
            expert_task_list = None

    else:
        train_task_list = [[args.robot, args.object, args.obstacle, args.objective]]
        experiment_name = f"{args.robot}_{args.object}_{args.obstacle}_{args.objective}_{args.hparam_id}"
        expert_task_list = None

    main(
        experiment_name=experiment_name,
        train_task_list=train_task_list,
        expert_task_list=expert_task_list,
        dataset_type=args.dataset_type,
        algo=args.algo,
        encoder=args.encoder,
        n_steps=args.n_steps,
    )
