from pathlib import Path

from hydra.experimental import compose, initialize
from omegaconf import DictConfig
import pytest

import hulc2

path = Path(hulc2.__file__).parent
config_files = [x.stem for x in (path.parent / "conf").glob("*.yaml")]


@pytest.mark.parametrize("config_name", config_files)
def test_cfg(config_name: str) -> None:
    with initialize(config_path="../conf"):
        cfg = compose(
            config_name=config_name,
            overrides=[
                "trainer.gpus=-1",
                "datamodule=default",
                "datamodule/observation_space=lang_rgb_static_rel_act",
                "datamodule/transforms=play_basic",
            ],
        )
        assert isinstance(cfg, DictConfig)
