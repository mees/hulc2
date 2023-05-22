import os
import json
import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import logging
from hulc2.utils.utils import get_abspath
from hulc2.utils.img_utils import add_img_text, resize_pixel, get_transforms
from hulc2.affordance.utils.data_utils import split_by_percentage
from hulc2.affordance.datasets.transforms import NormalizeInverse, NormalizeVector, NormalizeVectorInverse

class PixeLabelDataLang(Dataset):
    def __init__(
        self,
        img_resize,
        data_dir,
        transforms,
        data_percent=1.0,
        split="training",
        cam="static",
        log=None,
        episodes_file="episodes_split.json",
        *args,
        **kwargs
    ):
        super(PixeLabelDataLang, self).__init__()
        self.n_classes = 1
        self.cam = cam
        self.split = split
        self.log = log
        self.data_dir = get_abspath(data_dir)
        _data_info = self.read_json(os.path.join(self.data_dir, episodes_file))

        # Make all models to evaluate on same data
        _data_percent = 1.0 if split == "validation" else data_percent

        self.data = self._get_split_data(_data_info, split, cam, _data_percent)
        self.img_resize = img_resize
        _transforms_dct = get_transforms(transforms["validation"], img_resize[cam])
        self.transforms = _transforms_dct["transforms"]
        self.rand_shift = _transforms_dct["rand_shift"]
        _norm_vals = _transforms_dct["norm_values"]
        if(_norm_vals is not None):
            self.norm_inverse = NormalizeInverse(mean=_norm_vals.mean, std=_norm_vals.std)
        else:
            self.norm_inverse = None
        self.out_shape = self.get_shape(img_resize[cam])

        # Depth
        depth_norm_values = _data_info["norm_values"]["depth"]["%s_cam" % self.cam]
        self.depth_norm_values = depth_norm_values
        self.depth_norm = NormalizeVector([depth_norm_values["mean"]], [depth_norm_values["std"]])
        self.depth_norm_inversse = NormalizeVectorInverse(depth_norm_values["mean"], depth_norm_values["std"])

        # Excludes background
        # self.n_classes = _data_info["n_classes"] if cam == "static" else 1
        # self.n_classes = 2 if cam == "static" else 1
        self.resize = (self.img_resize[self.cam], self.img_resize[self.cam])
        self.cmd_log = logging.getLogger(__name__)
        self.cmd_log.info("Dataloader using shape: %s" % str(self.resize))

    def undo_normalize(self, x):
        if(self.norm_inverse is not None):
            x = self.norm_inverse(x)
        return x

    def read_json(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        return data

    def get_shape(self, in_size):
        test_tensor = torch.zeros((3, in_size, in_size))
        test_tensor = self.transforms(test_tensor)
        return test_tensor.shape  # C, H, W

    def _get_split_data(self, data, split, cam, data_percent):
        split_data = []
        split_episodes = list(data[split].keys())

        ## Hack to test overfit dataset
        if data_percent < 1.0:
            new_data = split_by_percentage(self.data_dir, data, data_percent)
        else:
            new_data = data

        print("%s episodes: %s" % (split, str(split_episodes)))
        for ep in split_episodes:
            split_files = ["%s/%s" % (ep, file) for file in new_data[split][ep]["%s_cam" % cam]]
            split_data.extend(split_files)
        print("%s images: %d" % (split, len(split_data)))
        return split_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # directions: optical flow image in middlebury color

        episode, filename = os.path.split(self.data[idx].replace("\\", "/"))
        data = np.load(self.data_dir + "/%s/data/%s_cam/%s.npz" % (episode, self.cam, filename))

        # Images are stored in BGR
        # tcp_world = data['tcp_pos_world_frame']
        old_shape = data["frame"].shape[:2]
        frame = data["frame"]
        orig_frame = torch.from_numpy(frame).permute(2, 0, 1)  # C, W, H
        frame = self.transforms(orig_frame.float())

        # Aff mask
        # data["centers"] = (label, x, y)
        center = resize_pixel(data["centers"][0, 1:], old_shape, self.resize)
        assert (center < self.resize).all(), "center is out of range, old_shape %s, resize %s, old_center %s, new_center %s" % (str(old_shape), str(self.resize), str(data["centers"][0, 1:]), str(center))
        # mask = np.zeros(self.resize)
        # mask[center[0], center[1]] = 1

        # Apply rand shift
        if(self.rand_shift is not None):
            frame, center = self.rand_shift({"img": frame,
                                             "center": center})

        # Select a language annotation
        annotations = [i.item() for i in data["lang_ann"]]
        assert len(annotations) > 0, "no language annotation in %s" % self.data[idx]
        lang_ann = np.random.choice(annotations).item()

        task = data["task"].tolist()
        inp = {"img": frame,  # RGB
               "lang_goal": lang_ann,
               "orig_frame": orig_frame.float() / 255}
        
        # Cam point in -z direction, but depth should be positive
        tcp_cam = data['tcp_pos_cam_frame']
        depth = tcp_cam[-1] * -1  
        norm_depth = self.depth_norm(torch.tensor(depth)).detach().cpu().item()

        # CE Loss requires mask in form (B, H, W)
        labels = {"task": task,
                  "p0": center,
                  "depth": depth,
                  "normalized_depth": norm_depth,
                  "tcp_pos_world_frame": data["tcp_pos_world_frame"],
                  "tetha0": []}
        return inp, labels


@hydra.main(config_path="../../config", config_name="train_affordance")
def get_imgs(cfg):
    data = PixeLabelDataLang(split="training", log=None, **cfg.aff_detection.dataset)
    loader = DataLoader(data, num_workers=1, batch_size=1, pin_memory=True)
    print("minibatches {}".format(len(loader)))

    cm = plt.get_cmap("jet")
    colors = cm(np.linspace(0, 1, data.n_classes))
    for b_idx, b in enumerate(loader):
        # RGB
        inp, labels = b
    
        # Imgs to numpy
        inp_img = data.undo_normalize(inp["img"]).detach().cpu().numpy()
        inp_img = (inp_img[0] * 255).astype("uint8")

        orig_img = inp["orig_frame"]
        frame = orig_img[0].detach().cpu().numpy()
        frame = (frame * 255).astype("uint8")
        frame = np.transpose(frame, (1, 2, 0))

        frame = cv2.resize(frame, inp["img"].shape[-2:])
        if frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        out_img = frame.copy()
        for label in range(0, data.n_classes):
            color = colors[label]
            color[-1] = 0.3
            color = tuple((color * 255).astype("int32"))

            # Draw center
            center_px = labels["p0"][0].numpy().squeeze()
            y, x = center_px[0].item(), center_px[1].item()
            out_img = cv2.drawMarker(
                out_img,
                (x, y),
                (0, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2,
                line_type=cv2.LINE_AA,
            )

        out_img = cv2.resize(out_img, (200, 200), interpolation=cv2.INTER_CUBIC)
        out_img = add_img_text(out_img, inp["lang_goal"][0],
                               font_scale=0.6,
                               thickness=1)
        out_img = out_img[:, :, ::-1]
        if(cfg.save_viz):
            file_dir = "./imgs"
            os.makedirs(file_dir, exist_ok=True)
            filename = os.path.join(file_dir, "frame_%04d.png" % b_idx)
            cv2.imwrite(filename, out_img)
        cv2.imshow("img", out_img)
        cv2.waitKey(1)

def main(cfg):
    data = PixeLabelDataLang(split="training", log=None, **cfg.aff_detection.dataset)
    loader = DataLoader(data, num_workers=1, batch_size=1, pin_memory=True)
    print("minibatches {}".format(len(loader)))

    cm = plt.get_cmap("jet")
    colors = cm(np.linspace(0, 1, data.n_classes))
    for b_idx, b in enumerate(loader):
        # RGB
        inp, labels = b
    
        # Imgs to numpy
        inp_img = data.undo_normalize(inp["img"]).detach().cpu().numpy()
        inp_img = (inp_img[0] * 255).astype("uint8")
        transformed_img = np.transpose(inp_img, (1, 2, 0)).copy()

        orig_img = inp["orig_frame"]
        frame = orig_img[0].detach().cpu().numpy()
        frame = (frame * 255).astype("uint8")
        frame = np.transpose(frame, (1, 2, 0))

        frame = cv2.resize(frame, inp["img"].shape[-2:])
        if frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        out_img = frame.copy()
        for label in range(0, data.n_classes):
            color = colors[label]
            color[-1] = 0.3
            color = tuple((color * 255).astype("int32"))

            # Draw center
            center_px = labels["p0"][0].numpy().squeeze()
            y, x = center_px[0].item(), center_px[1].item()
            transformed_img = cv2.drawMarker(
                transformed_img,
                (x, y),
                (0, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2,
                line_type=cv2.LINE_AA,
            )

        transformed_img = cv2.resize(transformed_img, (400, 400), interpolation=cv2.INTER_CUBIC)
        out_img = cv2.resize(out_img, (400, 400), interpolation=cv2.INTER_CUBIC)

        out_img = np.hstack([out_img, transformed_img])
        # Prints the text.
        out_img = add_img_text(out_img, inp["lang_goal"][0])

        out_img = out_img[:, :, ::-1]
        if(cfg.save_viz):
            file_dir = "./imgs"
            os.makedirs(file_dir, exist_ok=True)
            filename = os.path.join(file_dir, "frame_%04d.png" % b_idx)
            cv2.imwrite(filename, out_img)
        cv2.imshow("img", out_img)
        cv2.waitKey(0)


if __name__ == "__main__":
    get_imgs()
