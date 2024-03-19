import os
import hydra
from hydra.utils import get_original_cwd

import d3rlpy

from composuite import make
import torch

from utils.data_utils import get_task_list, get_partial_task_list
from utils.model_utils import (
    load_model,
    get_latest_model_path,
    create_trainer,
    try_get_load_path,
)
from algos.cp_iql import create_cp_encoderfactory

# fmt: off
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# fmt: on

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

GLOBAL_SUBTASK_KWARGS = {
    "has_renderer": False,
    "has_offscreen_renderer": False,
    "reward_shaping": True,
    "use_camera_obs": False,
    "use_task_id_obs": True,
    "env_horizon": 500,
}

GLOBAL_STEP_COUNTER = 0


def modified_reset(gym_env):
    original_reset = gym_env.reset

    def reset_wrapper(*args, **kwargs):
        global GLOBAL_STEP_COUNTER
        GLOBAL_STEP_COUNTER = 0

        obs, _ = original_reset(*args, **kwargs)
        return obs

    gym_env.reset = reset_wrapper


def modified_step(gym_env):
    original_step = gym_env.step

    def step_wrapper(*args, **kwargs):
        global GLOBAL_STEP_COUNTER
        GLOBAL_STEP_COUNTER += 1

        obs, rew, done, _, info = original_step(*args, **kwargs)

        if GLOBAL_STEP_COUNTER % 500 == 0:
            info["TimeLimit.truncated"] = True

        return obs, rew, done, info

    gym_env.step = step_wrapper


@hydra.main(config_path="_configs", config_name="finetune")
def main(cfg):
    # ${dataset.type}/${dataset.split}/${algo}/${dataset.seed}
    cfg.base_path = (
        cfg.base_path
        if os.path.isabs(cfg.base_path)
        else os.path.join(get_original_cwd(), cfg.base_path)
    )
    path = os.path.join(
        cfg.base_path,
        cfg.dataset.type,
        cfg.dataset.split,
        cfg.exp,
        cfg.algo,
        str(cfg.dataset.seed),
    )
    logger.info(f"Finetuning from {path}")

    hydra_path = os.path.join(path, ".hydra", "config.yaml")
    # load omegaconf from old run
    old_config_path = os.path.join(hydra_path)
    # old_cfg = omegaconf.OmegaConf.load(old_config_path)
    logger.info(f"Loaded old config from {old_config_path}")

    # load the task list
    _, train_task_list, expert_task_list, test_task_list = get_task_list(
        (
            cfg.dataset.task_list_path
            if os.path.isabs(cfg.dataset.task_list_path)
            else os.path.join(get_original_cwd(), cfg.dataset.task_list_path)
        ),
        cfg.dataset.type,
        cfg.dataset.split,
        cfg.dataset.holdout_elem,
        cfg.dataset.seed,
    )
    logger.info(
        f"Found train task list of length {len(train_task_list)} and test task list of length {len(test_task_list)}"
    )
    if expert_task_list:
        logger.info(f"Found expert task list of length {len(expert_task_list)}")

    if cfg.dataset.partial.use:
        if cfg.dataset.split == "compositional":
            logger.warning(
                "Careful, you specified compositional training but partial loading. "
                + "You may not get any expert tasks."
            )
        task_list = train_task_list + test_task_list
        _, test_task_list = get_partial_task_list(
            task_list, cfg.dataset.partial.remove_elems, cfg.dataset.partial.n_tasks
        )

    # sample a random test_task
    robot, obj, obst, subtask = test_task_list[cfg.task_id]
    logger.info(f"Finetuning on {robot}, {obj}, {obst}, {subtask}")

    # create the environment
    subtask_kwargs = {
        "robot": robot,
        "obj": obj,
        "obstacle": obst,
        "task": subtask,
    }

    env = make(**subtask_kwargs, **GLOBAL_SUBTASK_KWARGS)
    modified_reset(env)
    modified_step(env)
    logger.info(f"Environment created: {env}")

    load_path = try_get_load_path(
        os.path.join(get_original_cwd(), cfg.base_path),
        cfg.dataset.type,
        cfg.dataset.split,
        cfg.exp,
        cfg.algo,
        cfg.dataset.seed,
    )
    _, model_path = get_latest_model_path(load_path)
    logger.info(f"Attempting to load model from {model_path} for {cfg.algo}")

    trainer_kwargs = {}
    if cfg.algo == "cp_iql":
        trainer_kwargs["actor_encoder_factory"] = create_cp_encoderfactory()
        trainer_kwargs["critic_encoder_factory"] = create_cp_encoderfactory(
            with_action=True, output_dim=1
        )
        trainer_kwargs["value_encoder_factory"] = create_cp_encoderfactory(
            with_action=False, output_dim=1
        )

    if cfg.algo == "cp_bc":
        trainer_kwargs["encoder_factory"] = create_cp_encoderfactory()

    trainer = create_trainer(cfg.algo, trainer_kwargs)
    trainer = load_model(trainer, model_path, env=env)
    logger.info(f"Loaded model from {model_path} for {cfg.algo}")

    # finetune the model
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=cfg.n_steps, env=env)
    logger.info(f"Finetuning for {cfg.n_steps} steps")
    trainer.fit_online(
        env,
        buffer,
        n_steps=cfg.n_steps,
        n_steps_per_epoch=500,
        update_start_step=cfg.update_start_step,
        save_interval=20,
    )


if __name__ == "__main__":
    main()
