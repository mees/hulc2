import importlib
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Dict, List, Union

import git
import hydra
import numpy as np
from omegaconf import OmegaConf
import pytorch_lightning
from pytorch_lightning.utilities.cloud_io import load as pl_load
import torch
import tqdm

logger = logging.getLogger(__name__)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def initialize_pretrained_weights(model, cfg):
    pretrain_chk = pl_load(format_sftp_path(Path(cfg.pretrain_chk)), map_location=lambda storage, loc: storage)
    batch_size = model.plan_recognition.position_embeddings.weight.shape[0]
    weight = "plan_recognition.position_embeddings.weight"
    pretrain_chk["state_dict"][weight] = pretrain_chk["state_dict"][weight][:batch_size]
    if "pretrain_exclude_pr" in cfg and cfg.pretrain_exclude_pr:
        for key in list(pretrain_chk["state_dict"].keys()):
            if key.startswith("plan_recognition"):
                del pretrain_chk["state_dict"][key]
    model.load_state_dict(pretrain_chk["state_dict"], strict=False)


def get_git_commit_hash(repo_path: Path) -> str:
    repo = git.Repo(search_parent_directories=True, path=repo_path.parent)
    assert repo, "not a repo"
    changed_files = [item.a_path for item in repo.index.diff(None)]
    if changed_files:
        print("WARNING uncommitted modified files: {}".format(",".join(changed_files)))
    return repo.head.object.hexsha


def get_checkpoints_for_epochs(experiment_folder: Path, epochs: Union[List, str]) -> List:
    if isinstance(epochs, str):
        epochs = epochs.split(",")
        epochs = list(map(int, epochs))
    ep = lambda s: int(s.stem.split("=")[1])
    return [chk for chk in get_all_checkpoints(experiment_folder) if ep(chk) in epochs]


def get_all_checkpoints(experiment_folder: Path) -> List:
    if experiment_folder.is_dir():
        checkpoint_folder = experiment_folder / "saved_models"
        if checkpoint_folder.is_dir():
            checkpoints = sorted(Path(checkpoint_folder).iterdir(), key=lambda chk: chk.stat().st_mtime)
            if len(checkpoints):
                return [chk for chk in checkpoints if chk.suffix == ".ckpt"]
    return []


def get_last_checkpoint(experiment_folder: Path) -> Union[Path, None]:
    # return newest checkpoint according to creation time
    checkpoints = get_all_checkpoints(experiment_folder)
    if len(checkpoints):
        return checkpoints[-1]
    return None


def save_executed_code() -> None:
    print(hydra.utils.get_original_cwd())
    print(os.getcwd())
    shutil.copytree(
        os.path.join(hydra.utils.get_original_cwd(), "hulc2"),
        os.path.join(hydra.utils.get_original_cwd(), f"{os.getcwd()}/code/hulc2"),
    )


def info_cuda() -> Dict[str, Union[str, List[str]]]:
    return {
        "GPU": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        # 'nvidia_driver': get_nvidia_driver_version(run_lambda),
        "available": str(torch.cuda.is_available()),
        "version": torch.version.cuda,
    }


def info_packages() -> Dict[str, str]:
    return {
        "numpy": np.__version__,
        "pyTorch_version": torch.__version__,
        "pyTorch_debug": str(torch.version.debug),
        "pytorch-lightning": pytorch_lightning.__version__,
        "tqdm": tqdm.__version__,
    }


def nice_print(details: Dict, level: int = 0) -> List:
    lines = []
    LEVEL_OFFSET = "\t"
    KEY_PADDING = 20
    for k in sorted(details):
        key = f"* {k}:" if level == 0 else f"- {k}:"
        if isinstance(details[k], dict):
            lines += [level * LEVEL_OFFSET + key]
            lines += nice_print(details[k], level + 1)
        elif isinstance(details[k], (set, list, tuple)):
            lines += [level * LEVEL_OFFSET + key]
            lines += [(level + 1) * LEVEL_OFFSET + "- " + v for v in details[k]]
        else:
            template = "{:%is} {}" % KEY_PADDING
            key_val = template.format(key, details[k])
            lines += [(level * LEVEL_OFFSET) + key_val]
    return lines


def print_system_env_info():
    details = {
        "Packages": info_packages(),
        "CUDA": info_cuda(),
    }
    lines = nice_print(details)
    text = os.linesep.join(lines)
    return text


def get_portion_of_batch_ids(percentage: float, batch_size: int) -> np.ndarray:
    """
    Select percentage * batch_size indices spread out evenly throughout array
    Examples
    ________
     >>> get_portion_of_batch_ids(percentage=0.5, batch_size=32)
     array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
     >>> get_portion_of_batch_ids(percentage=0.2, batch_size=32)
     array([ 0,  5, 10, 16, 21, 26])
     >>> get_portion_of_batch_ids(percentage=0.01, batch_size=64)
     array([], dtype=int64)
    """
    num = int(batch_size * percentage)
    if num == 0:
        return np.array([], dtype=np.int64)
    indices = np.arange(num).astype(float)
    stretch = batch_size / num
    indices *= stretch
    return np.unique(indices.astype(np.int64))


def format_sftp_path(path):
    """
    When using network mount from nautilus, format path
    """
    if path.as_posix().startswith("sftp"):
        uid = os.getuid()
        path = Path(f"/run/user/{uid}/gvfs/sftp:host={path.as_posix()[6:]}")
    return path


def get_abspath(path_str):
    path_str = os.path.expanduser(path_str)
    if not os.path.isabs(path_str):
        hydra_cfg = hydra.utils.HydraConfig().cfg
        if hydra_cfg is not None:
            cwd = hydra.utils.get_original_cwd()
        else:
            cwd = os.getcwd()
        path_str = os.path.join(cwd, path_str)
        path_str = os.path.abspath(path_str)
    return path_str


def get_aff_model(train_folder, model_name, img_resize=None, eval=True):
    hydra_run_dir = get_abspath(train_folder)
    logger.info("loading aff model from %s" % hydra_run_dir)
    hydra_cfg_path = os.path.join(hydra_run_dir, ".hydra/config.yaml")
    if os.path.exists(hydra_cfg_path):
        run_cfg = OmegaConf.load(hydra_cfg_path)
    else:
        print("path does not exist %s" % hydra_cfg_path)
        return None, None

    # Load model
    model = load_aff_model(
        hydra_run_dir, model_name, transforms=run_cfg.aff_detection.dataset.transforms["validation"], eval=eval
    )
    return model, run_cfg


def load_aff_model(hydra_run_dir, model_name, eval=False, **kwargs):
    # Load model
    checkpoint_path = os.path.join(hydra_run_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_path, model_name)
    if os.path.isfile(checkpoint_path):
        aff_cfg = os.path.join(hydra_run_dir, ".hydra/config.yaml")
        if os.path.isfile(aff_cfg):
            train_cfg = OmegaConf.load(aff_cfg)
            _model_cfg = train_cfg.aff_detection
        if eval:
            _model_cfg.model_cfg.freeze_encoder.lang = True
            _model_cfg.model_cfg.freeze_encoder.aff = True
            _model_cfg.model_cfg.freeze_encoder.depth = True
        # Get class
        model_class = _model_cfg._target_.split(".")
        model_file = ".".join(_model_cfg._target_.split(".")[:-1])
        model_file = importlib.import_module(model_file)
        model_class = getattr(model_file, model_class[-1])

        # Instantiate
        model = model_class.load_from_checkpoint(checkpoint_path, strict=False, **kwargs)

        # Override default voting layer parameters
        if "hough_voting" in kwargs and "hough_voting" in model.model_cfg:
            model.init_voting_layer(kwargs["hough_voting"])
        logger.info("Model successfully loaded: %s" % checkpoint_path)
    else:
        logger.info("No checkpoint file found, loading untrained model: %s" % checkpoint_path)
    if eval:
        model.eval()
    return model
