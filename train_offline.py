import os
import json
import h5py
import numpy as np
from glob import glob

from tqdm import tqdm

import d3rlpy

import hydra
from hydra.utils import get_original_cwd

from utils.data_utils import get_task_list, get_partial_task_list
from algos.cp_iql import CompositionalIQL, create_cp_encoderfactory

from torch.cuda import is_available as cuda_available

# fmt: off
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# fmt: on

DEVICE = "cuda:0" if cuda_available() else "cpu"


def get_datasets(base_path, task_list, dataset_type):
    num_tasks = len(task_list)

    # preallocate memory
    observations = np.zeros((num_tasks * 1000000, 93), dtype=np.float32)
    actions = np.zeros((num_tasks * 1000000, 8), dtype=np.float32)
    rewards = np.zeros((num_tasks * 1000000,), dtype=np.float32)
    terminals = np.zeros((num_tasks * 1000000,), dtype=np.uint8)
    timeouts = np.zeros((num_tasks * 1000000,), dtype=np.uint8)

    logger.info(f"Loading {dataset_type} datafiles")
    for i, task in enumerate(tqdm(task_list, desc="Load task datafiles")):
        robot, obj, obst, subtask = task
        h5path = os.path.join(
            base_path, dataset_type, f"{robot}_{obj}_{obst}_{subtask}", "data.hdf5"
        )

        with h5py.File(h5path, "r") as dataset_file:
            for key in [
                "observations",
                "actions",
                "rewards",
                "terminals",
                "timeouts",
            ]:
                assert key in dataset_file, "Dataset is missing key %s" % key

            observations[i * 1000000 : (i + 1) * 1000000] = dataset_file[
                "observations"
            ][:]
            actions[i * 1000000 : (i + 1) * 1000000] = dataset_file["actions"][:]
            rewards[i * 1000000 : (i + 1) * 1000000] = dataset_file["rewards"][:]
            terminals[i * 1000000 : (i + 1) * 1000000] = dataset_file["terminals"][:]

            # timeouts should not happen when terminal, so set all timeouts
            # where terminal == 1 to 0
            timeouts[i * 1000000 : (i + 1) * 1000000] = dataset_file["timeouts"][:]
            timeouts[i * 1000000 : (i + 1) * 1000000][
                terminals[i * 1000000 : (i + 1) * 1000000] == 1
            ] = 0

    return observations, actions, rewards, terminals, timeouts

def create_trainer(algo, trainer_kwargs):
    if algo == "bc":
        trainer = d3rlpy.algos.BC(
            **trainer_kwargs,
            use_gpu=True if DEVICE == "cuda:0" else False,
        )
    elif algo == "cql":
        trainer = d3rlpy.algos.CQL(
            **trainer_kwargs,
            use_gpu=True if DEVICE == "cuda:0" else False,
        )
    elif algo == "iql":
        trainer = d3rlpy.algos.IQL(
            **trainer_kwargs,
            use_gpu=True if DEVICE == "cuda:0" else False,
        )
    elif algo == "cp_iql":
        trainer = CompositionalIQL(
            **trainer_kwargs,
            use_gpu=True if DEVICE == "cuda:0" else False,
        )
    else:
        raise NotImplementedError

    return trainer

def load_model(trainer, load_path, dataset):
    trainer.build_with_dataset(dataset)
    trainer.load_model(load_path)
    return trainer

def train_algo(exp_name, dataset, algo, run_kwargs, load_path=None):
    trainer = create_trainer(algo, run_kwargs["trainer_kwargs"])
    if load_path != "None" and load_path:
        # check if load path is absolute, else use get_original_cwd()
        load_path = (
            load_path
            if os.path.isabs(load_path)
            else os.path.join(get_original_cwd(), load_path)
        )
        # there might be several subdirectories to go through, 
        # iterate over them and find the latest model_{nr}.pt
        paths = glob(f"{load_path}/d3rlpy_logs/**")
        logger.info(f"Found {len(paths)} paths that might contain models")
        model_paths = []
        for path in paths:
            sub_paths = glob(f"{path}/**")
            model_paths.extend(
                [p for p in sub_paths if "model_" in p]
            )
        model_path = sorted(
            model_paths, key=lambda x: int(x.split("/")[-1].split("_")[-1].split(".")[0])
        )[-1]
        logger.info(f"Attempting to load model from {model_path} for {algo}")
        trainer = load_model(trainer, model_path, dataset)
        logger.info(f"Loaded model from {model_path} for {algo}")
        latest_step = int(model_path.split("/")[-1].split("_")[-1].split(".")[0])
    else:
        logger.info(f"No load path specified, training from scratch")
        latest_step = 0

    n_steps = 300000 - latest_step
    if n_steps <= 0:
        logger.info(f"Model already trained for {latest_step} steps")
        return

    logger.info(f"Training {algo} for {n_steps} steps")
    trainer.fit(
        dataset,
        n_steps_per_epoch=1000,
        n_steps=n_steps,
        experiment_name=exp_name,
        eval_episodes=dataset,
        **run_kwargs["fit_kwargs"],
    )

def try_get_load_path(base_path, dataset_type, split, exp, algo, seed):
    logger.info(
        "Supposed to reload, checking if there exists" + 
        "a path with the current configuration to load from")
    load_path = os.path.join(
        base_path, dataset_type, split, exp, algo, f"{seed}"
    )
    if not os.path.exists(load_path):
        logger.error(f"Path {load_path} does not exist")
        exit(1)
    assert os.path.exists(os.path.join(load_path, "d3rlpy_logs")), (
        f"Path {load_path} does not contain d3rlpy_logs"
    )
    # also assert that that path isnt empty
    assert len(glob(f"{load_path}/d3rlpy_logs/**")) > 0, (
        f"Path {load_path} does not contain any models"
    )
    logger.info(f"Found path {load_path} to load from")  

    return load_path

@hydra.main(config_path="_configs", config_name="offline")
def main(cfg):
    if cfg.reload:
        cfg.load_path = try_get_load_path(
            os.path.join(get_original_cwd(), "./_offline_training"), cfg.dataset.type, cfg.dataset.split, cfg.exp, cfg.algo, cfg.dataset.seed
        )

    task_list_path = (
        cfg.dataset.task_list_path
        if os.path.isabs(cfg.dataset.task_list_path)
        else os.path.join(get_original_cwd(), cfg.dataset.task_list_path)
    )
    exp_name, train_task_list, expert_task_list, test_task_list = get_task_list(
        task_list_path,
        cfg.dataset.type,
        cfg.dataset.split,
        cfg.dataset.holdout_elem,
        cfg.dataset.seed,
    )

    if cfg.dataset.partial.use:
        if cfg.dataset.split == "compositional":
            logger.warning(
                "Careful, you specified compositional training but partial loading. "
                + "You may not get any expert tasks."
            )
        task_list = train_task_list + test_task_list
        task_list, _ = get_partial_task_list(
            task_list, cfg.dataset.partial.remove_elems, cfg.dataset.partial.n_tasks
        )
    else:
        task_list = train_task_list
    logger.info(f"Task list contains these elements: {np.unique(task_list, axis=0)}")
    num_tasks = len(task_list)
    if expert_task_list:
        num_tasks += len(expert_task_list)

    logger.info(f"Training on {num_tasks} tasks")
    # check if data path is absolute, else use get_original_cwd()
    data_path = (
        cfg.dataset.dir
        if os.path.isabs(cfg.dataset.dir)
        else os.path.join(get_original_cwd(), cfg.dataset.dir)
    )

    observations, actions, rewards, terminals, timeouts = get_datasets(
        data_path,
        task_list,
        cfg.dataset.type,
    )

    logger.info(f"Added {len(task_list)}tasks to the dataset")
    if expert_task_list:
        (
            expert_observations,
            expert_actions,
            expert_rewards,
            expert_terminals,
            expert_timeouts,
        ) = get_datasets(
            data_path,
            expert_task_list,
            "expert",
        )
        observations = np.concatenate([observations, expert_observations])
        actions = np.concatenate([actions, expert_actions])
        rewards = np.concatenate([rewards, expert_rewards])
        terminals = np.concatenate([terminals, expert_terminals])
        timeouts = np.concatenate([timeouts, expert_timeouts])
        logger.info(f"Added {len(expert_task_list)} expert tasks to the dataset")

    mdp_dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        episode_terminals=timeouts,
    )

    run_kwargs = {
        "trainer_kwargs": {
            "batch_size": num_tasks * 256,
        },
        "fit_kwargs": {
            "save_interval": 10,
        },
    }

    if cfg.algo == "cp_iql":
        run_kwargs["trainer_kwargs"][
            "actor_encoder_factory"
        ] = create_cp_encoderfactory()
        run_kwargs["trainer_kwargs"]["critic_encoder_factory"] = (
            create_cp_encoderfactory(with_action=True, output_dim=1)
        )
        run_kwargs["trainer_kwargs"]["value_encoder_factory"] = (
            create_cp_encoderfactory(with_action=False, output_dim=1)
        )

    logger.info(f"Training {cfg.algo} on {exp_name}")
    train_algo(exp_name, mdp_dataset, cfg.algo, run_kwargs, cfg.load_path)


if __name__ == "__main__":
    main()
