import os
import json
import h5py
import numpy as np

from tqdm import tqdm

import d3rlpy

import hydra
from hydra.utils import get_original_cwd

from env_utils import get_task_list, get_partial_task_list
from cp_iql import CompositionalIQL, CompositionalEncoderFactory

from torch.cuda import is_available as cuda_available

# fmt: off
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# fmt: on

DEVICE = "cuda:0" if cuda_available() else "cpu"


def get_datasets(base_path, task_list, dataset_type):
    def get_keys(h5file):
        keys = []

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.append(name)

        h5file.visititems(visitor)
        return keys

    dataset_list = []
    for task in tqdm(task_list, desc="Load task datafiles"):
        robot, obj, obst, subtask = task
        h5path = os.path.join(
            base_path, dataset_type, f"{robot}_{obj}_{obst}_{subtask}", "data.hdf5"
        )

        data_dict = {}
        with h5py.File(h5path, "r") as dataset_file:
            for k in get_keys(dataset_file):
                try:  # first try loading as an array
                    data_dict[k] = dataset_file[k][:]
                except ValueError as e:  # try loading as a scalar
                    data_dict[k] = dataset_file[k][()]

        for key in [
            "observations",
            "actions",
            "rewards",
            "terminals",
            "timeouts",
        ]:
            assert key in data_dict, "Dataset is missing key %s" % key

        dataset_list.append(data_dict)
        break

    dataset_list = np.array(dataset_list)
    return dataset_list


def dictlist_to_flatlists(dictlist):
    observations = []
    actions = []
    rewards = []
    terminals = []
    timeouts = []

    for data_dict in dictlist:
        observations.append(data_dict["observations"])
        actions.append(data_dict["actions"])
        rewards.append(data_dict["rewards"])

        _term = data_dict["terminals"]
        _timeo = data_dict["timeouts"]
        # overwrite _timeo to be 0 if where _term is True
        previous_timeout_count = np.sum(data_dict["timeouts"])
        logger.info(f"Terminal count: {np.sum(_term)}")
        logger.info(f"Previous timeout count: {previous_timeout_count}")

        # This makes sure that no timeouts happen in terminal states
        # required for d3rlpy
        _timeo[_term.astype(bool)] = 0
        logger.info(f"New timeout count: {np.sum(_timeo)}")
        logger.info(
            f"Overwrote {np.sum(_timeo) - previous_timeout_count} timeouts to 0."
        )

        terminals.append(_term)
        timeouts.append(_timeo)

    for key in ["observations", "actions", "rewards", "terminals", "timeouts"]:
        assert len(dictlist) == len(locals()[key]), f"Length mismatch for {key}"

    observations = np.concatenate(observations, axis=0)
    actions = np.concatenate(actions, axis=0)
    rewards = np.concatenate(rewards, axis=0)
    terminals = np.concatenate(terminals, axis=0)
    timeouts = np.concatenate(timeouts, axis=0)

    assert (
        len(observations)
        == len(actions)
        == len(rewards)
        == len(terminals)
        == len(timeouts)
    ), "Length mismatch"
    assert observations.shape[1] == 93, "Observation shape mismatch"
    assert actions.shape[1] == 8, "Action shape mismatch"

    return observations, actions, rewards, terminals, timeouts


def create_cp_encoderfactory():
    obs_dim = 93
    act_dim = 8
    # fmt: off
    observation_positions = {'object-state': np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]), 'obstacle-state': np.array([14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]), 'goal-state': np.array([28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]), 'object_id': np.array([45, 46, 47, 48]), 'robot_id': np.array([49, 50, 51, 52]), 'obstacle_id': np.array([53, 54, 55, 56]), 'subtask_id': np.array([57, 58, 59, 60]), 'robot0_proprio-state': np.array([61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
       78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92])}
    # fmt: on

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

    fac = CompositionalEncoderFactory(
        encoder_kwargs,
    )

    return fac


def train_algo(exp_name, dataset, algo, run_kwargs):
    if algo == "bc":
        trainer = d3rlpy.algos.BC(
            **run_kwargs["trainer_kwargs"],
            use_gpu=True if DEVICE == "cuda:0" else False,    
        )
    elif algo == "cql":
        trainer = d3rlpy.algos.CQL(**run_kwargs["trainer_kwargs"],
            use_gpu=True if DEVICE == "cuda:0" else False,
        )
    elif algo == "iql":
        trainer = d3rlpy.algos.IQL(**run_kwargs["trainer_kwargs"],
            use_gpu=True if DEVICE == "cuda:0" else False,
        )
    elif algo == "cp_iql":
        trainer = CompositionalIQL(**run_kwargs["trainer_kwargs"],
            use_gpu=True if DEVICE == "cuda:0" else False,
        )
    else:
        raise NotImplementedError

    trainer.fit(
        dataset,
        n_steps_per_epoch=1000,
        n_steps=300000,
        experiment_name=exp_name,
        eval_episodes=dataset,
        **run_kwargs["fit_kwargs"],
    )


@hydra.main(config_path="_configs", config_name="offline")
def main(cfg):
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
    task_list = (
        train_task_list + expert_task_list if expert_task_list else train_task_list
    )

    if cfg.dataset.partial.use:
        if cfg.dataset.split == "compositional":
            logger.warning(
                "Careful, you specified compositional training but partial loading. "
                + "You may not get any expert tasks."
            )
        task_list = task_list + test_task_list
        task_list, _ = get_partial_task_list(
            task_list, cfg.dataset.partial.remove_elems, cfg.dataset.partial.n_tasks
        )
    logger.info(f"Training on {len(task_list)} tasks")
    logger.info(f"Task list contains these elements: {np.unique}")

    # check if data path is absolute, else use get_original_cwd()
    data_path = (
        cfg.dataset.dir
        if os.path.isabs(cfg.dataset.dir)
        else os.path.join(get_original_cwd(), cfg.dataset.dir)
    )
    dataset_list = get_datasets(
        data_path,
        task_list,
        cfg.dataset.type,
    )
    dataset = dictlist_to_flatlists(dataset_list)

    mdp_dataset = d3rlpy.dataset.MDPDataset(
        observations=dataset[0],
        actions=dataset[1],
        rewards=dataset[2],
        terminals=dataset[3],
        episode_terminals=dataset[4],
    )

    run_kwargs = {
        "trainer_kwargs": {
            "batch_size": len(dataset_list) * 256,
        },
        "fit_kwargs": {
            "save_interval": 10,
        },
    }

    if cfg.algo == "cp_iql":
        run_kwargs["trainer_kwargs"][
            "actor_encoder_factory"
        ] = create_cp_encoderfactory()

    logger.info(f"Training {cfg.algo} on {exp_name}")
    train_algo(exp_name, mdp_dataset, cfg.algo, run_kwargs)


if __name__ == "__main__":
    main()
