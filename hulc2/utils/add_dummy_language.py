import argparse
from pathlib import Path
from typing import Dict

import numpy as np
from tqdm import tqdm

TRAINING_DIR: str = "training"
VAL_DIR: str = "validation"


def main(input_params: Dict) -> None:
    dataset_root_str = input_params["dataset_root"]
    abs_datasets_dir = Path(dataset_root_str)
    val_ep_start_end_ids = np.load(abs_datasets_dir / VAL_DIR / "ep_start_end_ids.npy")
    train_ep_start_end_ids = np.load(abs_datasets_dir / TRAINING_DIR / "ep_start_end_ids.npy")
    train_lang_folder = abs_datasets_dir / TRAINING_DIR / "lang_annotations"
    train_lang_folder.mkdir(parents=True, exist_ok=True)
    train_file_name = "auto_lang_ann.npy"
    val_lang_folder = abs_datasets_dir / VAL_DIR / "lang_annotations"
    val_lang_folder.mkdir(parents=True, exist_ok=True)
    val_file_name = "embeddings.npy"
    collected_data_val: Dict = {
        "language": {"ann": [], "task": [], "emb": []},
        "info": {"episodes": [], "indx": []},
    }
    collected_data_train: Dict = {
        "language": {"ann": [], "task": [], "emb": []},
        "info": {"episodes": [], "indx": []},
    }
    # this is a hack for debugging purposes
    collected_data_val["info"]["indx"] = val_ep_start_end_ids
    collected_data_val["language"]["ann"] = ["pick up the banana"] * val_ep_start_end_ids.shape[0]
    lang_emb = np.zeros(384, dtype=np.float32)
    lang_emb[-1] = 1.0
    collected_data_val["language"]["emb"] = [lang_emb] * val_ep_start_end_ids.shape[0]
    collected_data_val["language"]["emb"] = np.expand_dims(np.stack(collected_data_val["language"]["emb"], axis=0), 1)

    collected_data_train["info"]["indx"] = train_ep_start_end_ids
    collected_data_train["language"]["ann"] = ["pick up the banana"] * train_ep_start_end_ids.shape[0]
    collected_data_train["language"]["emb"] = [lang_emb] * train_ep_start_end_ids.shape[0]
    collected_data_train["language"]["emb"] = np.expand_dims(
        np.stack(collected_data_train["language"]["emb"], axis=0), 1
    )

    np.save(val_lang_folder / val_file_name, collected_data_val)
    np.save(train_lang_folder / train_file_name, collected_data_train)
    np.save(val_lang_folder / train_file_name, collected_data_val)

    # for directory in [TRAINING_DIR, VAL_DIR]:
    #     glob_generator = abs_datasets_dir.glob(directory + "/*.npz")
    #     file_names = [x for x in glob_generator if x.is_file()]
    #     all_file.extend(file_names)
    # all_file.sort()
    # for current in tqdm(all_file, total=len(all_file)):
    #     data_cur = np.load(current, allow_pickle=True)
    #     lang_emb = np.zeros(384, dtype=np.float32)
    #     lang_emb[-1] = 1.0
    #     np.savez_compressed(
    #         current,
    #         data_cur,
    #         language=lang_emb,
    #     )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="data", help="directory where raw dataset is allocated")
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)
