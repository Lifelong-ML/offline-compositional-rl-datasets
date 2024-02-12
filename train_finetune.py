import os
import numpy as np
import hydra
from hydra.utils import get_original_cwd
from glob import glob

import d3rlpy
import omegaconf

from robosuite import make
import torch

from env_utils import get_task_list, get_partial_task_list, CompoSuiteGymnasiumWrapper

# fmt: off
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# fmt: on

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GLOBAL_SUBTASK_KWARGS = {
    "has_renderer": False,
    "has_offscreen_renderer": False,
    "reward_shaping": True,
    "use_camera_obs": False,
    "use_task_id_obs": True,
    "horizon": 500,
}


def load_model(latest_path, algo):
    # find the latest model, model_{nr}.pt, and load it
    paths = glob(f"{latest_path}/**")
    logger.info(f"Found {len(paths)} paths in {latest_path}")
    model_paths = [p for p in paths if "model_" in p]
    model_path = sorted(
        model_paths, key=lambda x: int(x.split("/")[-1].split("_")[-1].split(".")[0])
    )[-1]

    logger.info(f"Attempting to load model from {model_path} for {algo}")
    if algo == "iql":
        model = d3rlpy.load_learnable(model_path)
    else:
        raise NotImplementedError(f"Model loading for {algo} not implemented yet.")

    logger.info(f"Loaded model number {model_path}")
    return model


def get_test_task_list(
    task_list_path, dataset_type, split, holdout_elem, seed, remove_elems
):
    # get the correct task list
    task_list_path = (
        task_list_path
        if os.path.isabs(task_list_path)
        else os.path.join(get_original_cwd(), task_list_path)
    )
    _, train_task_list, expert_task_list, test_task_list = get_task_list(
        task_list_path,
        dataset_type,
        split,
        holdout_elem,
        seed,
    )
    task_list = (
        train_task_list + expert_task_list if expert_task_list else train_task_list
    )
    if split == "compositional":
        logger.warning(
            "Careful, you specified compositional training but partial loading. "
            + "You may not get any expert tasks."
        )
    task_list = task_list + test_task_list
    _, test_task_list = get_partial_task_list(
        task_list,
        remove_elems,
    )

    return test_task_list


@hydra.main(config_path="_configs", config_name="finetune")
def main(cfg):
    # ${dataset.type}/${dataset.split}/${algo}/${dataset.seed}
    cfg.base_path = (
        cfg.base_path
        if os.path.isabs(cfg.base_path)
        else os.path.join(get_original_cwd(), cfg.base_path)
    )
    path = os.path.join(
        cfg.base_path, cfg.dataset_type, cfg.split, cfg.algo, str(cfg.seed)
    )

    # find the last pathby sorting the last part of the path
    lowest_path = os.path.join(path, "d3rlpy_logs")
    paths = glob(f"{os.path.join(lowest_path, '**')}")
    latest_path = sorted(paths, key=lambda x: x.split("/")[-1])[-1]

    hydra_path = os.path.join(path, ".hydra", "config.yaml")
    # load omegaconf from old run
    old_config_path = os.path.join(hydra_path)
    old_cfg = omegaconf.OmegaConf.load(old_config_path)
    logger.info(f"Loaded old config from {old_config_path}")

    # load the task list
    test_task_list = get_test_task_list(
        old_cfg.dataset.task_list_path,
        old_cfg.dataset.type,
        old_cfg.dataset.split,
        old_cfg.dataset.holdout_elem,
        old_cfg.dataset.seed,
        old_cfg.dataset.partial.remove_elems,
    )
    logger.info(f"Loaded test task list with {len(test_task_list)} tasks")

    # sample a random test_task
    np.random.seed(cfg.seed)
    task_idx = np.random.randint(0, len(test_task_list))
    robot, obj, obst, subtask = test_task_list[task_idx]
    logger.info(f"Finetuning on {robot}, {obj}, {obst}, {subtask}")

    # create the environment
    subtask_kwargs = {
        "robots": robot,
        "object_type": obj,
        "obstacle": obst,
        "env_name": subtask + "Subtask",
    }

    env = make(**GLOBAL_SUBTASK_KWARGS, **subtask_kwargs)
    env = CompoSuiteGymnasiumWrapper(env)
    logger.info(f"Environment created: {env}")

    model = load_model(latest_path, old_cfg.algo)

    # finetune the model
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=cfg.n_steps, env=env)
    logger.info(f"Finetuning for {cfg.n_steps} steps")
    model.fit_online(
        env,
        buffer,
        n_steps=cfg.n_steps,
        n_steps_per_epoch=500,
        update_start_step=cfg.update_start_step,
        with_timestamp=True,
        save_interval=10,
    )


if __name__ == "__main__":
    main()
