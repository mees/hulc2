import os
import sqlite3
from sqlite3 import Error
import hydra
import numpy as np
from pip import main
from tqdm import tqdm
from .utils.utils import read_tasks
import math
from copy import deepcopy


def create_connection(db_file):
    """create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn


def get_annotations(conn, ignore_empty_tasks=True):
    """
    Query all rows in the lang_ann table
    :param conn: the Connection object
    :return:
    """

    sql_query = """
    SELECT lang_ann.task, lang_ann.color_x, lang_ann.color_y, sequences.start_frame, sequences.end_frame
    FROM lang_ann INNER JOIN sequences ON lang_ann.seq_id=sequences.seq_id
    """
    if ignore_empty_tasks:
        sql_query += '''WHERE lang_ann.task!="no_task"'''
    
    sql_query += ";"
    cur = conn.cursor()
    cur.execute(sql_query)

    rows = cur.fetchall()

    return rows


def filename_to_idx(filename):
    return int(filename.split("_")[-1][:-4])


def overwrite_colors(task, color_x, color_y):
    task = task.replace('[x]', color_x)
    task = task.replace('[y]', color_y)
    return task


def reduce_15hz(data, cfg):
    ''' Original data was played at 30 hz -> downsample to half the speed'''
    new_data = deepcopy(data)

    for idx, (start, end) in enumerate(data["info"]["indx"]):
        new_data["info"]["indx"][idx] = (math.ceil(start/2), math.floor(end/2))
    return new_data

def reduce_15hz_repeated(data, cfg):
    new_data = {
        "language": {"ann": [], "task": [], "emb": []},
        "info": {"episodes": [], "indx": []},
    }  # type: typing.Dict

    orig_path = hydra.utils.get_original_cwd()
    data_dir = os.path.join(orig_path, cfg.dataset_dir)
    start_end_ids_file = os.path.join(data_dir, "ep_start_end_ids.npy")
    # Play data episodes
    start_end_ids = np.load(start_end_ids_file, allow_pickle=True)

    lang_sequences = np.array(data["info"]["indx"])
    for play_start, play_end in start_end_ids:
        cond = np.logical_and(lang_sequences[:, 0] >= play_start, lang_sequences[:, 1] <= play_end)
        inside_ep = np.where(cond)[0]
        for idx, (start, end) in enumerate(lang_sequences[inside_ep]):
            # Language sequences are repeated
            for i in range(2):
                new_data["language"]["ann"].append(data["language"]["ann"][idx])
                new_data["language"]["task"].append(data["language"]["task"][idx])
                new_data["language"]["emb"].append(data["language"]["emb"][idx])

            # First sequence: even
            new_start = ((start - play_start) // 2) + play_start
            new_end = ((end - play_start) // 2) + play_start
            new_end = min(play_end, new_end)
            new_data["info"]["indx"].append((new_start, new_end))

            # Second sequence: odd
            new_start =  ((start - play_start) // 2) + ((play_end + play_start) // 2)
            new_end = ((end - play_start) // 2) + ((play_end + play_start) // 2)
            new_end = min(play_end, new_end)
            new_data["info"]["indx"].append((new_start, new_end))

    return new_data

@hydra.main(config_path="config", config_name="retrieve_data")
def main(cfg):
    data = {
        "language": {"ann": [], "task": [], "emb": []},
        "info": {"episodes": [], "indx": []},
    }  # type: typing.Dict

    # Get language model
    nlp_model = hydra.utils.instantiate(cfg.lang_model)

    # Get database connection
    db_path = hydra.utils.get_original_cwd()
    db_file = os.path.join(db_path, cfg.database_path)

    # If database is somewhere else, please modify
    # db_file = "/mnt/ssd_shared/Users/Jessica/Documents/Thesis_ssd/datasets/database.db"

    conn = create_connection(db_file)
    # Load annotations
    task_annotations = read_tasks()

    # Get database data
    rows = get_annotations(conn, cfg.ignore_empty_tasks)
    for task, color_x, color_y, start_fr, end_fr in tqdm(rows):
        if ('[x]' in task and color_x != "") or ('[y]' in task and color_y != ""):
            posible_annotations = task_annotations[task]
            ann_idx = np.random.randint(len(posible_annotations))
            ann = overwrite_colors(posible_annotations[ann_idx], color_x, color_y)
            task = overwrite_colors(task, color_x, color_y)
            data["language"]["ann"].append(ann)
            data["language"]["task"].append(task)
            emb = nlp_model(ann).permute(1,0).cpu().numpy()
            data["language"]["emb"].append(emb)
        else:
            # No task
            data["language"]["ann"].append(task)
            data["language"]["task"].append(task)
            data["language"]["emb"].append(np.zeros(5))

        start_idx = int(start_fr)
        end_idx = int(end_fr)
        data["info"]["indx"].append((start_idx, end_idx))

    # Save lang ann for original data
    root_dir = hydra.utils.get_original_cwd()
    save_path = os.path.join(root_dir, cfg.save_path)

    # Specific directory
    save_folder = os.path.join(save_path, nlp_model.name)
    os.makedirs(save_folder, exist_ok=True)
    save_file = os.path.join(save_folder, "auto_lang_ann.npy")
    np.save(save_file, data)

    # Downsampling: save lang ann
    # dirs = ["15hz", "15hz_repeated"]
    # reduce_methods = [reduce_15hz, reduce_15hz_repeated]
    # for folder, reduce_method in zip(dirs, reduce_methods):
    #     reduced_data = reduce_method(data, cfg)
    #     save_folder = os.path.join(save_path, folder)
    #     # Specific
    #     save_folder = os.path.join(save_folder, nlp_model.name)
    #     os.makedirs(save_folder, exist_ok=True)
    #     save_file = os.path.join(save_folder, "auto_lang_ann.npy")
    #     np.save(save_file, reduced_data)


if __name__ == "__main__":
    main()
