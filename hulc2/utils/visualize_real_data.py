import glob
import os

import cv2
import numpy as np
import tqdm


def normalize_depth(img):
    img_mask = img == 0
    istats = (np.min(img[img > 0]), np.max(img))
    imrange = (img.astype("float32") - istats[0]) / (istats[1] - istats[0])
    imrange[img_mask] = 0
    imrange = 255.0 * imrange
    imsz = imrange.shape
    nchan = 1
    if len(imsz) == 3:
        nchan = imsz[2]
    imgcanvas = np.zeros((imsz[0], imsz[1], nchan), dtype="uint8")
    imgcanvas[0 : imsz[0], 0 : imsz[1]] = imrange.reshape((imsz[0], imsz[1], nchan))
    return imgcanvas


# Ger valid numpy files with raw data
def get_files(path, extension, recursive=False):
    if not os.path.isdir(path):
        print("path does not exist: %s" % path)
    search_str = "/*.%s" % extension if not recursive else "**/*.%s" % extension
    files = glob.glob(path + search_str)
    if not files:
        print("No *.%s files found in %s" % (extension, path))
    files.sort()
    return files


def viz_data(data_dir):
    """Visualize teleop data recorded with Panda robot and check actions are valid"""
    files = get_files(data_dir, "npz", recursive=True)  # Sorted files
    # Remove camera calibration npz from iterable files
    files = [f for f in files if "camera_info.npz" not in f]

    for idx, filename in enumerate(tqdm.tqdm(files)):
        data = np.load(filename, allow_pickle=True)
        if data is None:
            continue  # Skip file

        new_size = (400, 400)
        for key in ["rgb_static", "depth_static", "rgb_gripper", "depth_gripper"]:
            img = cv2.resize(data[key], new_size)
            if "rgb" in key:
                cv2.imshow(key, img[:, :, ::-1])
            else:
                img2 = normalize_depth(img)
                img2 = cv2.applyColorMap(img2, cv2.COLORMAP_JET)
                cv2.imshow(key, img2)
        cv2.waitKey(1)


if __name__ == "__main__":
    data_dir = "/tmp/test_dataset"
    viz_data(data_dir)
