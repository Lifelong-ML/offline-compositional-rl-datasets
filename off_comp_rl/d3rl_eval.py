import json
import os
import gc
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np

import torch.nn as nn
from torch import from_numpy, no_grad

from offline_env_with_envs import MTLOfflineCompoSuiteEnv
from utils import load_model, load_compositional_model

TASK_LIST_PATH = "_TO_SET_"


def rollout_envs(env, model: nn.Module, num_steps: int, num_trajs: int, save_path: str):
    """Rollout a fixed number of trajectories of length num_steps using
    the given model and environment.

    Args:
        env (gym.Env): Environment to rollout.
        model (nn.Module): Model to rollout.
        num_steps (int, optional): Number of steps to rollout. Defaults to 500.
        num_trajs (int, optional): Number of trajectories to rollout. Defaults to 1.
        save_path (str, optional): Path to save the rollout data. Defaults to None.

    Returns:
        float: Average sum of rewards across all trajectories.
        float: Average success rate across all trajectories.
    """
    obs = env.reset()

    all_rewards = []
    all_successes = np.zeros((num_trajs, len(obs[0])))
    for i in tqdm(range(num_trajs)):
        curr_rewards = []
        for s in tqdm(range(num_steps)):
            if len(obs.shape) > 2:
                obs = obs[0]
            with no_grad():
                action = model.predict(from_numpy(obs).cuda())
            obs, reward, _, _ = env.step(list(action))

            all_successes[i, :] = np.logical_or(
                all_successes[i, :], np.squeeze(reward == 1.0, axis=1)
            )

            # Save the rewards per env
            with open(save_path, "a") as f:
                f.write(f"{i},{s}")
                for r in reward:
                    f.write(f",{r[0]}")
                f.write("\n")

            curr_rewards.append(reward)
        all_rewards.append(curr_rewards)
        obs = env.reset()
    return np.mean(np.sum(all_rewards, axis=1)), np.mean(all_successes)


def load_tasklist(dataset_split: str, data_seed: int, holdout_elem: str = None):
    """Load the task list for the given dataset split and data seed.

    Args:
        dataset_split (str): Dataset split to load.
        data_seed (int): Data seed to load.
        holdout_elem (str, optional): Holdout element to load. Only used
            for the holdout split. Defaults to None.

    Returns:
        list: List of training tasks.
    """
    if holdout_elem:
        task_list_path = os.path.join(
            TASK_LIST_PATH, f"{dataset_split}/{holdout_elem}_split_{data_seed}.json"
        )
    else:
        task_list_path = os.path.join(
            TASK_LIST_PATH, f"{dataset_split}/split_{data_seed}.json"
        )

    with open(task_list_path, "r") as f:
        json_file = json.load(f)
        train_task_list = json_file["train"]
        test_task_list = json_file["test"]

    return train_task_list, test_task_list


def parse_args():
    """Parse the command line arguments."""
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
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
    parser.add_argument("--holdout-elem", type=str, default=None)
    parser.add_argument("--encoder", default="mlp")
    parser.add_argument("--model-nr", type=int, default=300000)
    parser.add_argument("--algo", type=str, default="iql")

    args = parser.parse_args()
    return args


def main():
    """Main function. Loads the model and task list and rolls out the
    environment. Saves the results in a csv file and prints the average
    sum of rewards and success rate.

    Raises:
        NotImplementedError: If the encoder is not supported.
    """
    args = parse_args()
    np.random.seed(args.data_seed)

    train_task_list, test_task_list = load_tasklist(
        args.dataset_split, args.data_seed, args.holdout_elem
    )

    if args.dataset_split == "holdout":
        built_paths = os.path.join(
            args.path,
            f"{args.dataset_type}_{args.holdout_elem}_{args.dataset_split}_{args.data_seed}_*",
        )
    else:
        built_paths = os.path.join(
            args.path, f"{args.dataset_type}_{args.dataset_split}_{args.data_seed}_*"
        )
    built_path = glob(built_paths)[0]

    if args.encoder == "mlp":
        model = load_model(built_path, number=args.model_nr, algo=args.algo)
    elif args.encoder == "compositional":
        model = load_compositional_model(built_path, number=args.model_nr)
    else:
        raise NotImplementedError("No such encoder.")

    offline_kwargs = {"ref_min_score": 0, "ref_max_score": 500, "dataset_url": ""}

    rewards = []
    successes = []
    for name, tl in zip(["train", "test"], [train_task_list, test_task_list]):
        # create a file for each task list to save results in
        # where the headline contains the step and a column for each task
        save_path = os.path.join(built_path, f"results_{name}_perstep.csv")
        with open(save_path, "w") as f:
            f.write("traj,step")
            for task in tl:
                task_string = f"{task[0]}_{task[1]}_{task[2]}_{task[3]}"
                f.write(f",{task_string}")
            f.write("\n")

        env = MTLOfflineCompoSuiteEnv(
            task_list=tl, **({"offline_kwargs": offline_kwargs})
        )
        avg_cum_reward, avg_success = rollout_envs(env, model, save_path=save_path)
        rewards.append(avg_cum_reward)
        successes.append(avg_success)

        del env
        gc.collect()

    print("Train reward", rewards[0])
    print("Test reward", rewards[1])

    print("Train success", successes[0])
    print("Test success", successes[1])


if __name__ == "__main__":
    main()
