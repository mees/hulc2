from typing import DefaultDict

import numpy as np


def get_split_data(play_start_end_ids, data_percent, lang_data=None):
    start_end_ids = np.array(play_start_end_ids)
    cumsum = np.cumsum([e - s for s, e in play_start_end_ids])

    n_samples = int(cumsum[-1] * data_percent)
    max_idx = min(n_samples, cumsum[-1]) if n_samples > 0 else cumsum[-1]
    indices = [0]
    for i in range(len(cumsum) - 1):
        if cumsum[i] <= max_idx:
            indices.append(i + 1)

    # Valid play-data start_end_ids episodes
    start_end_ids = [start_end_ids[i] for i in indices]
    diff = cumsum[indices[-1]] - n_samples
    start_end_ids[-1][-1] = start_end_ids[-1][-1] - diff

    # Only add frames w/lang that are inside selected non-lang frames
    if lang_data is not None:
        lang_data = get_split_lang_sequences(start_end_ids, lang_data)
    return np.array(start_end_ids), lang_data


def get_split_lang_sequences(start_end_ids, lang_data, asarray=True):
    split_lang_data = {
        "language": {"ann": [], "task": [], "emb": []},
        "info": {"episodes": [], "indx": []},
    }
    # Language annotated episodes(64 frames)
    # keys = [(start_i, end_i), ...]
    keys = np.array([idx for idx in lang_data["info"]["indx"]])
    for start, end in start_end_ids:
        # Check if language annotated episode frames(64) are part of frames selected for non-language annotated frames(play data episodes).
        # i.e. Check that both language annotated and non-language come frome the same data
        cond = np.logical_and(keys[:, 0] >= start, keys[:, 1] <= end)
        inside_ep = np.where(cond)[0]

        # If lang-annotated ep is inside selected play-data ep copy selected ep
        for i in inside_ep:
            split_lang_data["language"]["ann"].append(lang_data["language"]["ann"][i])
            split_lang_data["language"]["task"].append(lang_data["language"]["task"][i])
            split_lang_data["language"]["emb"].append(lang_data["language"]["emb"][i])
            split_lang_data["info"]["indx"].append(lang_data["info"]["indx"][i])

    split_lang_data["language"]["emb"] = np.array(split_lang_data["language"]["emb"])
    return split_lang_data
