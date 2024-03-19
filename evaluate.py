import os
import gc
import hydra
from hydra.utils import get_original_cwd
from tqdm import tqdm
from functools import partial

from composuite import make
import torch

from utils.data_utils import get_task_list, get_partial_task_list
from utils.model_utils import (
    load_model,
    get_latest_model_path,
    create_trainer,
    try_get_load_path,
)
from utils.env_utils import DummyVecEnv
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


def rollout_envs(env, model, num_steps: int, num_trajs: int, save_path: str):
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
    for i in tqdm(range(num_trajs)):
        for s in tqdm(range(num_steps)):
            with torch.no_grad():
                action = model.predict(torch.from_numpy(obs).to(DEVICE))
            assert action.shape == (obs.shape[0], env.action_space.shape[0])
            obs, reward, dones, _ = env.step(list(action))

            # Save the rewards per env
            with open(save_path, "a") as f:
                f.write(f"{i},{s}")
                for r in reward:
                    f.write(f",{r}")
                f.write("\n")

            # Reset the envs that are done
            for j, done in enumerate(dones):
                if done:
                    obs[j] = env.envs[j].reset()[0]

        obs = env.reset()
    return True


def evaluate_tasklist(
    task_list, trainer, model_path, algo, trainer_kwargs, n_steps, n_trajs, save_loc
):
    env_fns = [
        partial(
            make,
            robot=robot,
            obj=obj,
            obstacle=obst,
            task=subtask,
            **GLOBAL_SUBTASK_KWARGS,
        )
        for robot, obj, obst, subtask in task_list
    ]
    logger.info(f"Creating environment for {len(env_fns)} tasks")
    env = DummyVecEnv(env_fns)

    trainer = create_trainer(algo, trainer_kwargs)

    logger.info(f"Attempting to load model from {model_path} for {algo}")
    trainer = load_model(trainer, model_path, env=env.envs[0])
    logger.info(f"Loaded model from {model_path} for {algo}")

    logger.info(f"Saving returns to {save_loc}")
    with open(save_loc, "w") as f:
        f.write("traj,step")
        for task in task_list:
            task_string = f"{task[0]}_{task[1]}_{task[2]}_{task[3]}"
            f.write(f",{task_string}")
        f.write("\n")

    logger.info(f"Rolling out {n_trajs} trajectories of length {n_steps}")
    rollout_envs(env, trainer, n_steps, n_trajs, save_loc)
    del env
    gc.collect()

    return trainer


@hydra.main(config_path="_configs", config_name="evaluate")
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

    if cfg.task_id != -1:
        test_task_list = [test_task_list[cfg.task_id]]

    logger.info(
        f"After partial loading, train task list has length {len(train_task_list)}"
    )
    if expert_task_list:
        logger.info(
            f"After partial loading, expert task list has length {len(expert_task_list)}"
        )
    logger.info(
        f"After partial loading, test task list has length {len(test_task_list)}"
    )

    # load the model
    load_path = try_get_load_path(
        os.path.join(get_original_cwd(), cfg.base_path),
        cfg.dataset.type,
        cfg.dataset.split,
        cfg.exp,
        cfg.algo,
        cfg.dataset.seed,
    )
    _, model_path = get_latest_model_path(load_path)

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

    trainer = None
    # evaluate the model
    if cfg.get_train_results:
        logger.info("Evaluating train task list")
        trainer = evaluate_tasklist(
            train_task_list,
            trainer,
            model_path,
            cfg.algo,
            trainer_kwargs,
            cfg.n_steps,
            cfg.n_trajs,
            "train_returns.csv",
        )
    else:
        logger.info("get_train_results is False, skipping Train evaluation")

    if expert_task_list:
        logger.info("Evaluating expert task list")
        trainer = evaluate_tasklist(
            expert_task_list,
            trainer,
            model_path,
            cfg.algo,
            trainer_kwargs,
            cfg.n_steps,
            cfg.n_trajs,
            "expert_returns.csv",
        )
    else:
        logger.info("No expert task list found, skipping Expert evaluation")

    logger.info("Evaluating test task list")
    trainer = evaluate_tasklist(
        test_task_list,
        trainer,
        model_path,
        cfg.algo,
        trainer_kwargs,
        cfg.n_steps,
        cfg.n_trajs,
        "test_returns.csv",
    )


if __name__ == "__main__":
    main()
