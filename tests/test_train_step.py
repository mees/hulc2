import hydra
from hydra.experimental import compose, initialize
from omegaconf import DictConfig
import pytest
import pytorch_lightning as pl
from pytorch_lightning import seed_everything


@pytest.fixture(scope="module")
def lightning_module() -> pl.LightningModule:
    with initialize(config_path="../conf"):
        cfg = compose(
            config_name="config.yaml",
        )
        assert isinstance(cfg, DictConfig)
        assert isinstance(cfg.model, DictConfig)
        model = hydra.utils.instantiate(cfg.model)
        seed_everything(cfg.seed)
        return model


def test_training_step_real_data(lightning_module, trainer, play_datamodule):
    trainer.fit(lightning_module, datamodule=play_datamodule)


def test_training_step_dummy_data(lightning_module, trainer, random_datamodule):
    trainer.fit(lightning_module, datamodule=random_datamodule)
