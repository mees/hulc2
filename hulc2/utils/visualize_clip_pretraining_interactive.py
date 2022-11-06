import logging
from pathlib import Path

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm

from hulc2.evaluation.utils import imshow_tensor
from hulc2.utils.utils import format_sftp_path, get_last_checkpoint

"""This script will collect data snt store it with a fixed window size"""

logger = logging.getLogger(__name__)


def generate_single_seq_gif(seq_img, seq_length, imgs, idx, i, data, pred_lang):
    s, c, h, w = seq_img.shape
    seq_img = np.transpose(seq_img, (0, 2, 3, 1))
    print("Seq length: {}".format(s))
    print("From: {} To: {}".format(idx[0], idx[1]))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for j in range(seq_length):
        imgRGB = seq_img[j]
        imgRGB = cv2.resize(
            ((imgRGB - imgRGB.min()) / (imgRGB.max() - imgRGB.min()) * 255).astype(np.uint8), (500, 500)
        )
        # img = plt.imshow(imgRGB, animated=True)
        # text1 = plt.text(
        #     200, 200, f"t = {j}", ha="center", va="center", size=10, bbox=dict(boxstyle="round", ec="b", lw=2)
        # )
        img = cv2.putText(imgRGB, f"t = {j}", (350, 450), font, color=(0, 0, 0), fontScale=1, thickness=2)
        img = cv2.putText(img, f"prediction {pred_lang}", (50, 70), font, color=(0, 255, 0), fontScale=0.5, thickness=1)
        img = cv2.putText(
            img, f"{i}. {data['language']['ann'][i]}", (100, 20), font, color=(0, 0, 0), fontScale=0.5, thickness=1
        )[:, :, ::-1]

        # text = plt.text(
        #     100,
        #     20,
        #     f"{i}. {data['language']['ann'][i]}",
        #     ha="center",
        #     va="center",
        #     size=10,
        #     bbox=dict(boxstyle="round", ec="b", lw=2),
        # )
        if j == 0:
            for _ in range(25):
                imgs.append(img)
        imgs.append(img)
    return imgs


def get_sequence(dataset, start_idx, end_idx):
    seq_length = end_idx - start_idx
    dataset.max_window_size, dataset.min_window_size = seq_length, seq_length
    start = dataset.episode_lookup.index(start_idx)
    return dataset[start]


def get_clip_scores(model, obs, lang_emb, lang_instructions, tasks, obs_space):
    obs_filtered = {
        key: {key2: ob for key2, ob in obs_dict.items() if key2 in obs_space[key]}
        for key, obs_dict in obs.items()
        if key in obs_space.keys()
    }
    obs_filtered["robot_obs"] = obs_filtered["state_obs"]
    del obs_filtered["state_obs"]
    scores = model.clip_inference(obs_filtered, {"lang": lang_emb})
    scores = scores.cpu().numpy().squeeze()
    ranking = np.argsort(scores)[::-1]
    scores_sorted = np.sort(scores)[::-1]
    lang_instructions_sorted = [lang_instructions[i] for i in ranking]
    tasks_sorted = list(np.array(tasks)[ranking])
    return lang_instructions_sorted, scores_sorted, tasks_sorted


def plot_scores(
    train_pred_instructions,
    train_pred_scores,
    train_pred_tasks,
    val_pred_instructions,
    val_pred_scores,
    val_pred_tasks,
    gt_task,
    save_name="train",
):
    train_colors = ["#79afd5", "#d62728"]  # , "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]
    val_colors = ["#005a98", "#ce7e00"]  # , "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]
    train_color = [train_colors[int(task == gt_task)] for task in train_pred_tasks]
    val_color = [val_colors[int(task == gt_task)] for task in val_pred_tasks]

    scores = np.concatenate([train_pred_scores, val_pred_scores])
    instructions = train_pred_instructions + val_pred_instructions
    color = train_color + val_color
    scores, instructions, color = map(
        list, zip(*sorted(zip(scores, instructions, color), key=lambda t: t[0], reverse=True))
    )

    fig = plt.figure(figsize=(8, len(scores) // 5))
    ax = plt.subplot()

    normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    x = np.arange(len(scores))

    ax.barh(x, normalized_scores, color=color, height=0.5)
    ax.set_yticks(x)
    ax.set_yticklabels(labels=[name.replace("_", " ").capitalize() for name in instructions], fontsize=10, ha="right")
    ax.xaxis.set_tick_params(length=0)
    ax.invert_yaxis()
    # plt.legend(loc="upper right", prop={"size": 18})
    fig.savefig(f"/tmp/{save_name}.pdf", dpi=300, bbox_inches="tight")


def get_obs_window(obs, start, length):
    obs_window = {
        "rgb_obs": {key: imgs.unsqueeze(0).cuda()[:, start : start + length] for key, imgs in obs["rgb_obs"].items()},
        "depth_obs": {},
        "state_obs": {},
    }
    return obs_window


def visualize_clip(train_lang_data, dataset, models, val_lang_data, cfg, obs_spaces):
    val_lang_tasks = []
    val_lang_emb = []
    val_lang_instructions = []
    for val_task, val_instructions in cfg.val_instructions.items():
        val_lang_tasks.append(val_task)
        val_lang_emb.append(torch.from_numpy(val_lang_data[val_task]["emb"][0]).cuda())
        val_lang_instructions.append(list(val_lang_data[val_task]["ann"])[0])
    val_lang_emb = torch.cat(val_lang_emb)
    train_lang_instructions = list(set(train_lang_data["language"]["ann"]))
    train_lang_ids = [train_lang_data["language"]["ann"].index(instruction) for instruction in train_lang_instructions]
    train_lang_emb = torch.from_numpy(train_lang_data["language"]["emb"][train_lang_ids]).cuda().squeeze()
    train_lang_tasks = list(np.array(train_lang_data["language"]["task"])[train_lang_ids])

    i = 0
    test_seq_len = 32
    num_seq = len(train_lang_data["info"]["indx"])
    start_idx, end_idx = train_lang_data["info"]["indx"][i]
    j = 0
    rgb_obs = get_sequence(dataset, start_idx, end_idx)
    rgb_obs_window = get_obs_window(rgb_obs, j, test_seq_len)
    gt_task = train_lang_data["language"]["task"][i]
    while 1:
        imshow_tensor("start", rgb_obs_window["rgb_obs"]["rgb_static"][0, 0], wait=1, text=gt_task)
        imshow_tensor("middle", rgb_obs_window["rgb_obs"]["rgb_static"][0, test_seq_len // 2], wait=1)
        imshow_tensor("end", rgb_obs_window["rgb_obs"]["rgb_static"][0, -1], wait=1)
        k = cv2.waitKey(0) % 256
        if k == ord("a"):
            j -= 1
            j = np.clip(j, 0, end_idx - start_idx - test_seq_len)
            rgb_obs_window = get_obs_window(rgb_obs, j, test_seq_len)
        if k == ord("d"):
            j += 1
            j = np.clip(j, 0, end_idx - start_idx - test_seq_len)
            rgb_obs_window = get_obs_window(rgb_obs, j, test_seq_len)
        if k == ord("q"):
            i -= 1
            i = np.clip(i, 0, num_seq)
            start_idx, end_idx = train_lang_data["info"]["indx"][i]
            j = 0
            rgb_obs = get_sequence(dataset, start_idx, end_idx)
            rgb_obs_window = get_obs_window(rgb_obs, j, test_seq_len)
            gt_task = train_lang_data["language"]["task"][i]
        if k == ord("e"):
            i += 1
            i = np.clip(i, 0, num_seq)
            start_idx, end_idx = train_lang_data["info"]["indx"][i]
            j = 0
            rgb_obs = get_sequence(dataset, start_idx, end_idx)
            rgb_obs_window = get_obs_window(rgb_obs, j, test_seq_len)
            gt_task = train_lang_data["language"]["task"][i]
        if k == ord("r"):
            for model, name, obs_space in zip(models, ["32_64", "16_32"], obs_spaces):
                pred_instructions, pred_scores, pred_tasks = get_clip_scores(
                    model, rgb_obs_window, train_lang_emb, train_lang_instructions, train_lang_tasks, obs_space
                )
                val_pred_instructions, val_pred_scores, val_pred_tasks = get_clip_scores(
                    model, rgb_obs_window, val_lang_emb, val_lang_instructions, val_lang_tasks, obs_space
                )
                plot_scores(
                    pred_instructions,
                    pred_scores,
                    pred_tasks,
                    val_pred_instructions,
                    val_pred_scores,
                    val_pred_tasks,
                    gt_task,
                    name,
                )


def load_data(cfg):
    seed_everything(cfg.seed)
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=4)
    data_module.prepare_data()
    data_module.setup()
    dataset = data_module.train_dataloader()["vis"].dataset

    file_name = dataset.abs_datasets_dir / cfg.lang_folder / "auto_lang_ann.npy"
    task_embeddings = np.load(data_module.val_dir / cfg.lang_folder / "embeddings.npy", allow_pickle=True).reshape(-1)[
        0
    ]
    return np.load(file_name, allow_pickle=True).reshape(-1)[0], dataset, task_embeddings


def load_model(dir, checkpoint):
    train_cfg_path = format_sftp_path(Path(dir)) / ".hydra/config.yaml"
    train_cfg = OmegaConf.load(train_cfg_path)
    logger.info("Loading model from checkpoint.")
    model = hydra.utils.instantiate(train_cfg.model)
    # checkpoint = get_last_checkpoint(Path(dir))  # is not the last
    model = model.load_from_checkpoint(format_sftp_path(Path(checkpoint)))
    model.freeze()
    return model.cuda(), train_cfg.datamodule.observation_space


@hydra.main(config_path="../../conf", config_name="lang_ann.yaml")
def main(cfg: DictConfig) -> None:
    train_lang_data, dataset_obj, task_emb = load_data(cfg)
    model, obs_space = load_model(
        "sftp://kis2bat3.rz.ki.privat/home/hermannl/logs/2022-02-08/18-44-33_pretrain_no_lang_proj_32_64_auxwindow_8",
        checkpoint="sftp://kis2bat3.rz.ki.privat/home/hermannl/logs/2022-02-08/18-44-33_pretrain_no_lang_proj_32_64_auxwindow_8/saved_models/epoch=18.ckpt",
    )
    model2, obs_space2 = load_model(
        "sftp://kis2bat3.rz.ki.privat/home/hermannl/logs/2022-02-08/18-45-29_pretrain_no_lang_proj_16_32_auxwindow_4",
        checkpoint="sftp://kis2bat3.rz.ki.privat/home/hermannl/logs/2022-02-08/18-45-29_pretrain_no_lang_proj_16_32_auxwindow_4/saved_models/epoch=40.ckpt",
    )
    models = [model, model2]
    obs_spaces = [obs_space, obs_space2]

    visualize_clip(train_lang_data, dataset_obj, models, task_emb, cfg, obs_spaces)


if __name__ == "__main__":
    main()
