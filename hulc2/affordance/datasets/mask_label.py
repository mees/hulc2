import json
import logging
import os

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hulc2.affordance.datasets.transforms import NormalizeInverse
from hulc2.affordance.utils.utils import split_by_percentage
import hulc2.utils.flowlib as flowlib
from hulc2.utils.utils import add_img_text, get_abspath, get_transforms, overlay_flow, overlay_mask, resize_pixel


class MaskLabelLabelDataLang(Dataset):
    def __init__(
        self,
        img_resize,
        data_dir,
        transforms,
        radius,
        data_percent=1.0,
        split="training",
        cam="static",
        log=None,
        episodes_file="episodes_split.json",
        *args,
        **kwargs,
    ):
        super(MaskLabelLabelDataLang, self).__init__()
        self.cam = cam
        self.split = split
        self.log = log
        self.data_dir = get_abspath(data_dir)
        _data_info = self.read_json(os.path.join(self.data_dir, episodes_file))
        self.data = self._get_split_data(_data_info, split, cam, data_percent)
        self.img_resize = img_resize
        _transforms_dct = get_transforms(transforms[split], img_resize[cam])
        self.transforms = _transforms_dct["transforms"]
        self.rand_shift = _transforms_dct["rand_shift"]
        _norm_vals = _transforms_dct["norm_values"]
        if _norm_vals is not None:
            self.norm_inverse = NormalizeInverse(mean=_norm_vals.mean, std=_norm_vals.std)
        else:
            self.norm_inverse = None
        self.out_shape = self.get_shape()  # C H W
        self.pixel_indices = np.indices((self.out_shape[-2:]), dtype=np.float32).transpose(1, 2, 0)
        self.radius = radius[cam]

        # Excludes background
        self.n_classes = _data_info["n_classes"] if cam == "static" else 1
        self.resize = (self.img_resize[self.cam], self.img_resize[self.cam])
        self.cmd_log = logging.getLogger(__name__)
        self.cmd_log.info("Dataloader using shape: %s" % str(self.resize))

    def undo_normalize(self, x):
        if self.norm_inverse is not None:
            x = self.norm_inverse(x)
        return x

    def read_json(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        return data

    def get_shape(self):
        test_tensor = torch.zeros((3, 10, 10))
        test_tensor = self.transforms(test_tensor)
        return test_tensor.shape  # C, H, W

    def _get_split_data(self, data, split, cam, data_percent):
        split_data = []
        split_episodes = list(data[split].keys())

        new_data = split_by_percentage(self.data_dir, data, data_percent)

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
        old_shape = data["frame"].shape[:2]
        frame = data["frame"]
        orig_frame = torch.from_numpy(frame).permute(2, 0, 1)  # C, W, H
        frame = self.transforms(orig_frame.float())

        # Apply transforms to center px
        # data["centers"] = (label, x, y)
        center = data["centers"][0][1:]
        center = resize_pixel(center, old_shape, self.resize)

        # Apply rand shift
        if self.rand_shift is not None:
            frame, center = self.rand_shift({"img": frame, "center": center})

        # Get mask and direction vectors
        center_dirs, mask = self.label_directions(center)
        # mask = np.expand_dims(mask, 0) # 1, H, W

        # Select a language annotation
        annotations = [i.item() for i in data["lang_ann"]]
        assert len(annotations) > 0, "no language annotation in %s" % self.data[idx]
        lang_ann = np.random.choice(annotations).item()

        task = data["task"].tolist()
        inp = {"img": frame, "lang_goal": lang_ann, "orig_frame": orig_frame.float() / 255}  # RGB

        # CE Loss requires mask in form (B, H, W)
        labels = {"task": task, "p0": center, "affordance": torch.tensor(mask).long(), "center_dirs": center_dirs}
        return inp, labels

    def label_directions(self, center_px):
        """
        center_px: np.array(shape=(2,), dtype='int64') pixel index resized
        """
        n_classes = self.n_classes
        new_shape = self.pixel_indices.shape
        direction_labels = np.stack([np.ones(new_shape[:-1]), np.zeros(new_shape[:-1])], axis=-1).astype(
            np.float32
        )  # H, W, 2
        obj_mask = np.zeros(new_shape[:-1])  # H, W
        indices = self.pixel_indices
        r = self.radius

        # centers are already resized
        c = center_px
        label = self.n_classes - 1  # 0 or 1
        object_center_directions = (c - indices).astype(np.float32)
        object_center_directions = object_center_directions / np.maximum(
            np.linalg.norm(object_center_directions, axis=2, keepdims=True), 1e-10
        )

        # Add it to the labels
        new_mask = self.create_circle_mask(obj_mask, c, r=r)
        new_mask = self.tresh_np(new_mask, 100).squeeze()
        direction_labels[new_mask == 1] = object_center_directions[new_mask == 1]
        obj_mask = overlay_mask(new_mask, obj_mask, (255, 255, 255))

        direction_labels = torch.tensor(direction_labels).permute(2, 0, 1)
        return direction_labels.float(), obj_mask.astype("uint8")

    def create_circle_mask(self, img, px, r=10):
        xy_coords = px[::-1]
        mask = np.zeros((img.shape[0], img.shape[1], 1))
        color = [255, 255, 255]
        mask = cv2.circle(mask, xy_coords, radius=r, color=color, thickness=-1)
        return mask

    def tresh_np(self, img, threshold=100):
        new_img = np.zeros_like(img)
        idx = img > threshold
        new_img[idx] = 1
        return new_img


def test_dir_labels(hv, frame, aff_mask, center_dir):
    bool_mask = (aff_mask == 1).int().cuda()
    center_dir = center_dir.cuda()  # 1, 2, H, W
    initial_masks, num_objects, object_centers_padded = hv(bool_mask, center_dir.contiguous())

    initial_masks = initial_masks.cpu()
    object_centers_padded = object_centers_padded[0].cpu().permute((1, 0))
    for c in object_centers_padded:
        c = c.detach().cpu().numpy()
        u, v = int(c[1]), int(c[0])  # center stored in matrix convention
        frame = cv2.drawMarker(
            frame,
            (u, v),
            (0, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=5,
            line_type=cv2.LINE_AA,
        )
    cv2.imshow("hv_center", frame[:, :, ::-1])
    cv2.waitKey(1)
    return frame


@hydra.main(config_path="../../config", config_name="train_affordance")
def main(cfg):
    data = MaskLabelLabelDataLang(split="training", log=None, **cfg.aff_detection.dataset)
    loader = DataLoader(data, num_workers=1, batch_size=1, pin_memory=True)
    print("val minibatches {}".format(len(loader)))
    from hulc2.affordance.hough_voting import hough_voting as hv

    hv = hv.HoughVoting(**cfg.aff_detection.model.cfg.hough_voting)

    cm = plt.get_cmap("jet")
    colors = cm(np.linspace(0, 1, data.n_classes))
    for b_idx, b in enumerate(loader):
        # RGB
        inp, labels = b

        # Labels
        directions = labels["center_dirs"][0].detach().cpu().numpy()
        mask = labels["affordance"].detach().cpu().numpy()
        directions = np.transpose(directions, (1, 2, 0))
        flow_img = flowlib.flow_to_image(directions)  # RGB

        # Imgs to numpy
        inp_img = data.undo_normalize(inp["img"]).detach().cpu().numpy()
        inp_img = (inp_img[0] * 255).astype("uint8")
        inp_img = np.transpose(inp_img, (1, 2, 0)).copy()

        orig_img = inp["orig_frame"]
        frame = orig_img[0].detach().cpu().numpy()
        frame = (frame * 255).astype("uint8")
        frame = np.transpose(frame, (1, 2, 0))

        frame = cv2.resize(frame, inp["img"].shape[-2:])
        if frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Add flow + bin mask
        out_img = frame.copy()
        out_img = overlay_flow(flow_img, frame, mask * 255)

        # Draw center in transformed img
        center_px = labels["p0"][0].numpy().squeeze()
        y, x = center_px[0].item(), center_px[1].item()
        transformed_img = cv2.drawMarker(
            inp_img,
            (x, y),
            (0, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=2,
            line_type=cv2.LINE_AA,
        )

        hv_center_img = test_dir_labels(hv, out_img, labels["affordance"], labels["center_dirs"])

        transformed_img = cv2.resize(transformed_img, (400, 400), interpolation=cv2.INTER_CUBIC)
        hv_center_img = cv2.resize(hv_center_img, (400, 400), interpolation=cv2.INTER_CUBIC)
        out_img = np.hstack([hv_center_img, transformed_img])

        # Prints the text.
        out_img = add_img_text(out_img, inp["lang_goal"][0])
        out_img = out_img[:, :, ::-1]
        if cfg.save_viz:
            file_dir = "./imgs"
            os.makedirs(file_dir, exist_ok=True)
            filename = os.path.join(file_dir, "frame_%04d.png" % b_idx)
            cv2.imwrite(filename, out_img)
        cv2.imshow("img", out_img)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
