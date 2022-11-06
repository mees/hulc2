import hydra
from hydra.experimental import compose, initialize
from omegaconf import DictConfig
import pytest
import pytorch_lightning as pl
from pytorch_lightning import seed_everything


@pytest.fixture(scope="module")
def lightning_lang_module() -> pl.LightningModule:
    with initialize(config_path="../conf"):
        cfg = compose(
            config_name="config.yaml",
            overrides=[
                "datamodule/datasets=vision_lang",
                "model/language_goal=default",
                "model.language_goal.language_features=768",
            ],
        )
        assert isinstance(cfg, DictConfig)
        assert isinstance(cfg.model, DictConfig)
        model = hydra.utils.instantiate(cfg.model)
        seed_everything(cfg.seed)
        return model


def test_training_step_lang_real_data(lightning_lang_module, trainer, play_datamodule_lang):
    trainer.fit(lightning_lang_module, datamodule=play_datamodule_lang)
