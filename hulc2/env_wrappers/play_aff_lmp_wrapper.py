import logging
import os
from typing import Any, Dict, Tuple, Union

import gym
import torch
from calvin_env.calvin_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id

logger = logging.getLogger(__name__)


class PlayLMPWrapper(gym.Wrapper):
    def __init__(self, env, device, **kwargs):
        self.set_egl_device(device)
        super(PlayLMPWrapper, self).__init__(env)
        # _action_min = np.array(env_cfg.action_min)
        # _action_high = np.array(env_cfg.action_max)
        # self.action_space = spaces.Box(_action_min, _action_high)
        # self.observation_space_keys = env_cfg.observation_space
        # self.proprio_state = env_cfg.proprioception_dims
        self.device = device
        logger.info(f"Initialized PlayTableEnv for device {self.device}")

    @staticmethod
    def set_egl_device(device):
        if "EGL_VISIBLE_DEVICES" in os.environ:
            logger.warning("Environment variable EGL_VISIBLE_DEVICES is already set. Is this intended?")
        cuda_id = device.index if device.type == "cuda" else 0
        try:
            egl_id = get_egl_device_id(cuda_id)
        except EglDeviceNotFoundError:
            logger.warning(
                "Couldn't find correct EGL device. Setting EGL_VISIBLE_DEVICE=0. "
                "When using DDP with many GPUs this can lead to OOM errors. "
                "Did you install PyBullet correctly? Please refer to calvin env README"
            )
            egl_id = 0
        os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
        logger.info(f"EGL_DEVICE_ID {egl_id} <==> CUDA_DEVICE_ID {cuda_id}")

    def reset(
        self,
        reset_info: Dict[str, Any] = None,
        batch_idx: int = 0,
        seq_idx: int = 0,
        scene_obs: Any = None,
        robot_obs: Any = None,
    ) -> Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        if reset_info is not None:
            obs = self.env.reset(
                robot_obs=reset_info["robot_obs"][batch_idx, seq_idx],
                scene_obs=reset_info["scene_obs"][batch_idx, seq_idx],
            )
        elif scene_obs is not None or robot_obs is not None:
            obs = self.env.reset(scene_obs=scene_obs, robot_obs=robot_obs)
        else:
            obs = self.env.reset()
        

        return obs

    def get_info(self):
        return self.env.get_info()

    def get_obs(self):
        obs = self.env.get_obs()
        return obs
