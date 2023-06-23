# Discovering affordances from play data
The dataset will automatically be generated in /home/USER/datasets. To change this add the following line:
`paths.datasets=DATASETS_PATH`
The paths configuration is taken from [general_paths.yaml](../conf/paths/general_paths.yaml)

The expected folder structure is:
```
DATASETS_PATH/
 |_ PROCESSED_PLAY_DATASET_NAME/
 |  |_ training/
 |  |_ validation/
 |  |_ episodes_split.json
 |_ unprocessed/
    |_ PLAY_DATASET_NAME/
      |_ camera_info.npz
      |_ ep_start_end_ids.npy
      |_ statistics.yaml
      |_ split.json
      |_ calib/
      |_ lang_paraphrase-MiniLM-L3-v2/
hulc2/
 |__ calvin_env/
 |   |__ data/
 |__ hulc2/
 |__ conf/
  ...
```
but you can change the directory where `PROCESSED_PLAY_DATASET_NAME` is by specifying the `paths.datasets=PROCESSED_PLAY_DATASET_NAME` argument where `PROCESSED_PLAY_DATASET_NAME` is the absolute path of the directory containing the labeled play data.
## Generate affordance dataset
Before training the affordance model, the play data (`PLAY_DATASET_NAME`) needs to pass through the data_labeler_lang.py to be processed. This will generate the `PROCESSED_PLAY_DATASET_NAME` directory, which will be then used for training the affordance model

### Locally labeling the affordances
Modify PLAY_DATA_ABS_DIRECTORY to match the absolute path of where you're storing the play data, otherwise it will look for it under ../datasets/unprocessed/PLAY_DATASET_NAME as indicated above.

Because of how the play data is generated, this needs to be run separately for training and validation.

### Generating training affordance dataset
```
PLAY_DATASET_NAME="PLAY_DATASET_NAME/training"

python hulc2.affordance.dataset_creation.data_labeler_lang
dataset_name=$PLAY_DATASET_NAME \\
output_cfg.single_split=training \\
output_size.static=200 \\
output_size.gripper=84 \\
labeling=simulation_lang \\
mask_on_close=True \\
```

### Generating validation affordance dataset
```
PLAY_DATASET_NAME="PLAY_DATASET_NAME/validation"

python hulc2.affordance.dataset_creation.data_labeler_lang
dataset_name=$PLAY_DATASET_NAME \\
output_cfg.single_split=validation \\
output_size.static=200 \\
output_size.gripper=84 \\
labeling=simulation_lang \\
mask_on_close=True \\
```

For processing the real world dataset, use the flag `labeling=real_world_lang`.
### Merging the datasets
After generating the data for training and validation of the affordance model, we need to create a json using the merge_datasets.py script. This will generate a episodes_split.json file in the PLAY_DATASET_NAME directory.

The directories to merge are specified in [cfg_merge_dataset.yaml](../conf/affordance/cfg_merge_dataset.yaml). They can be relative or absolute paths. By default the script outputs to the parent of the first directory in the list of cfg_merge_dataset.yaml

```
python -m hulc2.affordance.dataset_creation.merge_datasets
```

## Training Affordance Model
### Running locally:
Modify DATASETS_PATH and DATASET_NAME according to where you're storing the data. Here we show an example

```
DATASETS_PATH="/home/datasets"
DATASET_NAME="real_world/500k_all_tasks_dataset_15hz"

python -m hulc2.affordance.train_affordance \
dataloader.batch_size=32 \
paths.datasets=$DATASETS_PATH \
dataset_name=$DATASET_NAME \
aff_detection.dataset.cam=static \
aff_detection=r3m \
aff_detection.model_cfg.freeze_encoder.lang=True \
aff_detection.model_cfg.freeze_encoder.aff=False \
aff_detection.dataset.data_percent=1.0 \
run_name=local
```

## Evaluation model-based + model-free with affordance
Script: [evaluate_policy.py](../hulc2/evaluation/evaluate_policy.py)

### Running on a cluster:
AFF_TRAIN_FOLDER is the hydra directory for the desired affordance model to use.

```
cd run_on_cluster
python slurm_eval.py -v hulc2 --dataset_path DATA_TO_EVAL_PATH \\
--train_folder POLICY_TRAIN_FOLDER \\
--checkpoint EPOCH_NR | NONE \\
--aff_train_folder AFF_TRAIN_FOLDER
```

When the "--checkpoint" argument is not specified,  the evaluation will run for all epochs in POLICY_TRAIN_FOLDER.

The results are stored in POLICY_TRAIN_FOLDER the following way:
- If the evaluation ran with an affordance model: POLICY_TRAIN_FOLDER/evaluacion/Hulc_Aff/date/time.
- If the evaluation ran using only Hulc: POLICY_TRAIN_FOLDER/evaluation/Hulc/date/time

Here we show an example.
```
python slurm_eval.py -v thesis --hours 23 -j hulc_eval_all --dataset_path /work/dlclarge2/meeso-lfp/calvin_data/task_D_D --train_folder /home/borjadi/logs/lfp/2022-07-13/23-15-01_gcbc_50 --aff_train_folder /work/dlclarge2/borjadi-workspace/logs/thesis/aff_ablation/2022-06-20/14-31-59_aff_ablation
```

Optional flags:
 - --save_viz (flag to save images of rollouts)
 - --cameras=high_res (flag to change camera cfg to save images)
