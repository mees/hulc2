import hydra
from hydra.experimental import compose, initialize
from omegaconf import DictConfig
import pytest
import pytorch_lightning as pl

from hulc2.datasets.random import RandomDataModule
from tests.datasets import TEST_CACHE_DIR


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="module")
def random_datamodule():
    data_module = RandomDataModule(batch_size=2)
    return data_module


@pytest.fixture(scope="module")
def play_datamodule():
    with initialize(config_path="../conf"):
        cfg = compose(
            config_name="config.yaml",
            overrides=[
                "datamodule.num_workers=0",
                "datamodule/datasets=vision_only",
                "datamodule/observation_space=lang_rgb_static_abs_act",
                "datamodule.datasets.vision_dataset.batch_size=2",
                "datamodule.root_data_dir=" + TEST_CACHE_DIR,
            ],
        )
        assert isinstance(cfg, DictConfig)
        data_module = hydra.utils.instantiate(cfg.datamodule)
        return data_module


@pytest.fixture(scope="module")
def play_datamodule_lang():
    with initialize(config_path="../conf"):
        cfg = compose(
            config_name="config.yaml",
            overrides=[
                "datamodule.num_workers=0",
                "datamodule/datasets=vision_lang",
                "datamodule/observation_space=lang_rgb_static_abs_act",
                "datamodule.datasets.lang_dataset.batch_size=2",
                "datamodule.datasets.vision_dataset.batch_size=2",
                "datamodule.datasets.lang_dataset.min_window_size=3",
                "datamodule.datasets.lang_dataset.max_window_size=6",
                "datamodule.datasets.lang_dataset.lang_folder=lang_annotations",
                "datamodule.root_data_dir=" + TEST_CACHE_DIR,
            ],
        )
        assert isinstance(cfg, DictConfig)
        data_module_lang = hydra.utils.instantiate(cfg.datamodule)
        return data_module_lang


@pytest.fixture(scope="module")
def trainer() -> pl.Trainer:
    trainer = pl.Trainer(fast_dev_run=True, logger=False)
    return trainer
