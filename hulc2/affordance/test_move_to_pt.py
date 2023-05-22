import hydra
from hulc2.env_wrappers.play_aff_lmp_wrapper import PlayLMPWrapper
import torch

@hydra.main(config_path="../../conf", config_name="cfg_calvin")
def main(cfg):
    # Load env
    env = hydra.utils.instantiate(cfg.env)
    env = PlayLMPWrapper(env, torch.device('cuda:0'))
    agent = hydra.utils.instantiate(cfg.agent,
                                    env=env,
                                    aff_cfg=cfg.aff_detection)
    obs = env.reset()

    captions = ["Lift the red block",
                "Stored the grasped block in the cabinet",
                "turn on the yellow light"]
    for caption in captions:  # n instructions
        # caption = "use the switch to turn on the light bulb" # input("Type an instruction \n")
        # caption = "open the drawer"
        # obs = env.reset()
        agent.reset(caption)
        if agent.model_free.lang_encoder is not None:
            goal = {"lang": [caption]}
        else:
            goal = agent.encode(caption)
        for j in range(cfg.max_timesteps):
            action = agent.step(obs, goal)
            obs, _, _, info = env.step(action)
        agent.save_dir["rollout_counter"] += 1
    agent.save_sequence_txt("sequence", captions)
    agent.save_sequence()

if __name__ == "__main__":
    main()
