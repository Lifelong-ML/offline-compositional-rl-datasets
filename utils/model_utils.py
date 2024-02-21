import os
from glob import glob

import d3rlpy
import torch
from algos.cp_iql import CompositionalIQL

from hydra.utils import get_original_cwd

# fmt: off
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# fmt: on

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def load_model(trainer, load_path, dataset=None, env=None):
    if dataset is not None:
        trainer.build_with_dataset(dataset)
    elif env is not None:
        trainer.build_with_env(env)
    else:
        raise ValueError("Need either dataset or env to build model")
    
    trainer.load_model(load_path)
    return trainer

def load_latest_model_from_path(trainer, load_path, algo, dataset=None, env=None):
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
        sub_model_paths = [p for p in sub_paths if "model_" in p]
        assert len(sub_model_paths) > 0, f"Path {path} does not contain any models"
        model_paths.extend(
            sub_model_paths
        )
    model_path = sorted(
        model_paths, key=lambda x: int(x.split("/")[-1].split("_")[-1].split(".")[0])
    )[-1]
    logger.info(f"Attempting to load model from {model_path} for {algo}")
    trainer = load_model(trainer, model_path, dataset, env)
    logger.info(f"Loaded model from {model_path} for {algo}")
    latest_step = int(model_path.split("/")[-1].split("_")[-1].split(".")[0])

    return latest_step, trainer

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