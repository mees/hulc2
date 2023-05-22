from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
import os

def add_img_text(img, text_label):
    font_scale = 0.6
    thickness = 2
    color = (0, 0, 0)
    im_w, im_h = img.shape[:2]
    x1, y1 = 10, 20
    (w, h), _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    out_img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1 + h), color, -1)
    out_img = cv2.putText(
        out_img,
        text_label,
        org=(x1, y1),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=(255, 255, 255),
        thickness=thickness,
    )
    return out_img

def main():
    # Please first run get_annotations to generate auto_lang_ann.npy
    lang_ann_path = Path(__file__).resolve().parents[1] / "annotations" / "lang_paraphrase-MiniLM-L3-v2" / "auto_lang_ann.npy"
    
    # Path where dataset is
    dataset_path = "/mnt/ssd_shared/Users/Jessica/Documents/Thesis_ssd/datasets/unprocessed/real_world/500k_all_tasks_dataset_15hz"

    annotations = np.load(lang_ann_path.resolve(), allow_pickle=True).item()
    indices = [317, 723, 22]
    for index in indices:
        idx = index - 1
        caption = annotations["language"]["ann"][idx]
        start_fr, end_fr = annotations["info"]["indx"][idx]
        for fr in range(start_fr, end_fr):
            frame_file = os.path.join(dataset_path, "episode_%07d.npz" % fr)
            step_file = np.load(frame_file)
            img = step_file["rgb_static"]
            w,h = img.shape[:2]
            img = cv2.resize(img, (h*3, w*3))
            img = add_img_text(img, caption)
            cv2.imshow("img", img[:, :, ::-1])
            cv2.waitKey(0)
        cv2.waitKey(1)

if __name__=="__main__":
    main()