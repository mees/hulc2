import logging
from pathlib import Path

import cv2
import hydra
from matplotlib.animation import ArtistAnimation
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm

from hulc2.utils.utils import get_last_checkpoint

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


def generate_all_seq_gifs(data, dataset, model, task_emb, cfg):
    imgs = []
    # fig = plt.figure()
    lang_list = []
    lang_text_list = []
    for i, (val_task, val_instructions) in enumerate(cfg.val_instructions.items()):
        lang_list.append(torch.from_numpy(task_emb[val_task]["emb"][0]).cuda())
        lang_text_list.append(task_emb[val_task]["ann"])
    lang_batch = torch.cat(lang_list)
    for i, idx in enumerate(tqdm(data["info"]["indx"][:20])):
        seq_length = idx[1] - idx[0]
        dataset.max_window_size, dataset.min_window_size = seq_length, seq_length
        start = dataset.episode_lookup.index(idx[0])
        img_cuda = dataset[start]["rgb_obs"]["rgb_static"].unsqueeze(0).cuda()
        seq_len = img_cuda.shape[1]
        max_len = 32
        mod = int(seq_len / max_len)
        for seq_slice in range(mod):
            img_cuda_slice = img_cuda[:, max_len * seq_slice : max_len * (seq_slice + 1), :, :, :]
            in_dict = {"rgb_obs": {"rgb_static": img_cuda_slice}, "depth_obs": {}, "robot_obs": {}}
            logits_per_image = model.clip_inference(in_dict, {"lang": lang_batch})
            idx_max = logits_per_image.argmax().item()
            pred_lang = lang_text_list[idx_max]
            # print("pred lang", pred_lang)
            seq_img = dataset[start]["rgb_obs"]["rgb_static"].numpy()
            # if 'lift' in data['language']['task'][i]:
            imgs = generate_single_seq_gif(
                seq_img[max_len * seq_slice : max_len * (seq_slice + 1), :, :, :],
                max_len,
                imgs,
                idx,
                i,
                data,
                pred_lang,
            )
    return imgs


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


def plot_and_save_gifs(imgs):
    # anim = ArtistAnimation(fig, imgs, interval=75)
    # plt.axis("off")
    # plt.title("Annotated Sequences")
    # plt.show()
    # anim.save("/tmp/summary_lang_anns.mp4", writer="ffmpeg", fps=15)
    print("plot_and_save_gifs")
    # video = cv2.VideoWriter("/tmp/summary_lang_anns.avi", cv2.VideoWriter_fourcc(*"XVID"), 15, (500, 500))
    for img in imgs:
        cv2.imshow("bla", img)
        cv2.waitKey(100)
    #     video.write(img)
    # video.release()


def generate_task_id(tasks):
    labels = list(sorted(set(tasks)))
    task_ids = [labels.index(task) for task in tasks]
    return task_ids


def visualize_embeddings(data, with_text=True):
    emb = data["language"]["emb"].squeeze()
    tsne_emb = TSNE(n_components=2, random_state=40, perplexity=20.0).fit_transform(emb)

    emb_2d = tsne_emb

    task_ids = generate_task_id(data["language"]["task"])

    cmap = ["orange", "blue", "green", "pink", "brown", "black", "purple", "yellow", "cyan", "red", "grey", "olive"]
    ids_in_legend = []
    for i, task_id in enumerate(task_ids):
        if task_id not in ids_in_legend:
            ids_in_legend.append(task_id)
            plt.scatter(emb_2d[i, 0], emb_2d[i, 1], color=cmap[task_id], label=data["language"]["task"][i])
            if with_text:
                plt.text(emb_2d[i, 0], emb_2d[i, 1], data["language"]["ann"][i])
        else:
            plt.scatter(emb_2d[i, 0], emb_2d[i, 1], color=cmap[task_id])
            if with_text:
                plt.text(emb_2d[i, 0], emb_2d[i, 1], data["language"]["ann"][i])
    plt.legend()
    plt.title("Language Embeddings")
    plt.show()


def load_model(dir):
    train_cfg_path = Path(dir) / ".hydra/config.yaml"
    train_cfg = OmegaConf.load(train_cfg_path)
    logger.info("Loading model from checkpoint.")
    model = hydra.utils.instantiate(train_cfg.model)
    # checkpoint = get_last_checkpoint(Path(dir))  # is not the last
    checkpoint = Path("/home/meeso/17-27-15_pretrain_clip_loss_static_lang_b128/saved_models/epoch=36.ckpt")
    model = model.load_from_checkpoint(checkpoint)
    model.freeze()
    return model.cuda()


@hydra.main(config_path="../../conf", config_name="lang_ann.yaml")
def main(cfg: DictConfig) -> None:
    data, dataset_obj, task_emb = load_data(cfg)
    model = load_model("/home/meeso/17-27-15_pretrain_clip_loss_static_lang_b128/")
    # visualize_embeddings(data)
    imgs = generate_all_seq_gifs(data, dataset_obj, model, task_emb, cfg)
    plot_and_save_gifs(imgs)


if __name__ == "__main__":
    main()
