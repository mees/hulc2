# Learning from play pipeline
## Process the real world data
Processing script to transform real world data into simulation format
- [process_real_data](../hulc2/utils/preprocess_real_data.py)
```
python process_real_data.py --dataset_root SRC_DATA_PATH --output_dir DEST_DATA_PATH
```

## Convert to 15 hz
Render teleop data in 15 hz with render_low_freq
- [render_low_freq](../hulc2/utils/render_low_freq.py)

```
python render_low_freq.py --dataset_root SRC_DATA_PATH --output_dir DEST_DATA_PATH
```

## Create dataset split
Generate a data split on split.json and compute statistics.yaml:
- [split_dataset.py](../hulc2/utils/split_dataset.py)

```
 python split_dataset.py --dataset_root FULL_DATA_PATH
```

If we modify the data split manually, then we need to re-compute statistics.yaml

## Generate auto_lang_ann (LangAnnotationApp):
- [get_annotations.py](../hulc2/scripts/get_annotations.py)

```
 python get_annotations.py dataset_dir=FULL_DATASET_DIR database_path=FULL_DATABASE_PATH
```

FULL_DATASET_DIR: Absolute path of directory where the preprocessed dataset is (output of process_real_data). This is, original data before reducing the frequency
FULL_DATABASE_PATH: Absolute path of direction where database storing annotations for dataset in FULL_DATASET_DIR is.

This script will produce multiple ouputs inside a folder named [annotations](../../LanguageAnnotationApp/annotations/) for the different types of dataset:
    - original 30hz in [annotations](../../LanguageAnnotationApp/annotations/)
    - reduced15hz in [15hz](../../LanguageAnnotationApp/annotations/15hz)
    - repeated15hz in [15hz_repeated](../../LanguageAnnotationApp/annotations/15hz_repeated/)

Copy the language annotation folder to the corresponding dataset.

## Train learning from play
- [training.py](../hulc2/training.py)

# Dataset folder content
Dataset should contain the following files/folders:
- statistics.yaml
- split.json
- ep_start_end_ids.npy
- .hydra/

If additionally using language:
- lang_[LANG_MODEL]/auto_lang_ann.npy


#  Affordance model and depth prediction
## Dataset generation

1. Discover the affordances to get the affordance pixel and depth labels:
script:
[data_labeler_lang.py](../hulc2/affordance/dataset_creation/data_labeler_lang.py)

config:
[cfg_datacollection.py](../conf/affordance/cfg_datacollection.yaml)

### Example: Simulation

```
conda activate hulc2
python -m hulc2.affordance.dataset_creation.data_labeler_lang
```

### Example: Real-World data


2. Get normalization values from the training set:
[find_norm_values.py](../hulc2/affordance/dataset_creation/find_norm_values.py)

## Training

### Train depth estimation module:
[train_depth.py](../hulc2/affordance/train_depth.py)

### Train the affordance model:
script:
[train_affordance.py](../hulc2/affordance/train_affordance.py)

configuration:
[train_affordance.yaml](../conf/affordance/train_affordance.yaml)

```
conda activate hulc2
python -m hulc2.affordance.train_affordance \
dataloader.batch_size=4 \
dataset_name=real_world/500k_all_tasks_dataset_15hz \
+aff_detection.dataset.episodes_file=episodes_split_mini.json \
aff_detection.dataset.cam=static \
aff_detection=r3m \
aff_detection.model_cfg.freeze_encoder.lang=True \
aff_detection.model_cfg.freeze_encoder.aff=False \
aff_detection.dataset.data_percent=1.0 \
run_name=test \
```

## Evaluation
### Visualize de-projected end effector position:
[test_move_to_pt.py](../hulc2/affordance/test_move_to_pt.py)

### Visualize affordance predictions:
[test_affordance.py](../hulc2/affordance/test_affordance.py)
configuration:
[test_affordance.yaml](../conf/affordance/test_affordance.yaml)

```
conda activate hulc2
python -m hulc2.affordance.test_affordance.py \
checkpoint.train_folder=OUTPUT_HYDRA_HYDRA_DIRECTORY/100p \
dataset_name=calvin_lang_MoCEndPt \
debug=True \
```
