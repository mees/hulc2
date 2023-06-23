# Training Affordance Model
The dataset will automatically be generated in /home/USER/datasets. To change this add the following line:
`paths.datasets=DATASETS_PATH`
The paths configuration is taken from [general_paths.yaml](./config/paths/general_paths.yaml)

The expected folder structure is:
```
datasets
 |_ PROCESSED_PLAY_DATASET_NAME
 |_ unprocessed
    |_ PLAY_DATASET_NAME
hulc2
 |__ calvin_env
     |__ data
 |__ hulc2
 |__ conf
  ...
```
but you can change the directory where `PROCESSED_PLAY_DATASET_NAME` is by specifying the `paths.datasets=PROCESSED_PLAY_DATASET_NAME` argument where `PROCESSED_PLAY_DATASET_NAME` is the absolute path of the directory containing the labeled play data.
## Generate affordance dataset
Before training the affordance model, the play data (`PLAY_DATASET_NAME`) needs to pass through the data_labeler_lang.py to be processed. This will generate the `PROCESSED_PLAY_DATASET_NAME` directory, which will be then used for training the affordance model

### Locally labeling the affordances
Modify PLAY_DATA_ABS_DIRECTORY to match the absolute path of where you're storing the play data.
```
PLAY_DATA_ABS_DIRECTORY="/home/datasets/some_play_data_dir"

python hulc2.affordance.dataset_creation.data_labeler_lang
play_data_dir=$PLAY_DATA_ABS_DIRECTORY \\
dataset_name=calvin_langDepthEndPt/training \\
output_cfg.single_split=training \\
output_size.static=200 \\
output_size.gripper=86 \\
labeling=simulation_lang \\
mask_on_close=True \\
```

### Training set
```
python cluster.py -v "hulc2" --train_file "../thesis/affordance/dataset_creation/data_labeler_lang.py" --no_clone -j labeling \\
paths.parent_folder=/home/USER/ \\
play_data_dir=PLAY_DATA_DIRECTORY \\
dataset_name=calvin_langDepthEndPt/training \\
output_cfg.single_split=training \\
output_size.static=200 \\
output_size.gripper=86 \\
labeling=simulation_lang \\
mask_on_close=True \\
```
### Validation set
```
python cluster.py -v "hulc2" --train_file "../hulc2/affordance/dataset_creation/data_labeler_lang.py" --no_clone -j labeling \\
paths.parent_folder=/home/USER/ \\
play_data_dir=/work/dlclarge2/meeso-lfp/calvin_data/task_D_D/validation \\
dataset_name=calvin_langDepthEndPt/validation \\
output_cfg.single_split=validation \\
output_size.static=200 \\
output_size.gripper=86 \\
labeling=simulation_lang \\
mask_on_close=True \\
```
### Merging the datasets
The directories to merge are specified in [cfg_merge_dataset.yaml](../conf/affordance/cfg_merge_dataset.yaml). They can be relative to the "thesis" directory or absolute paths. By default the script outputs to the parent of the first directory in the list of cfg_merge_dataset.yaml

```
python merge_datasets.py
```

## Find normalization values for depth prediction
Script: [find_norm_values.py](../hulc2/affordance/dataset_creation/find_norm_values.py)

### Running on a cluster:
```
cd run_on_cluster
python cluster.py -v "thesis" --train_file "../scripts/find_norm_values.py" --no_clone  -j norm_values --data_dir /DATASETS_PATH/calvin_langDepthEndPt/
```

## Train model
### Running locally:
Modify DATASETS_PATH and DATASET_NAME according to where you're storing the data.

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
### Running on a cluster:
```
cd run_on_cluster
python cluster.py -v "hulc2" --train_file "../train_affordance.py" --no_clone \\
-j aff_model \\
paths.parent_folder=~/ \\
run_name=WANDB_RUN_NAME \\
aff_detection=VISUAL_LANGENC_LABEL_TYPE \\
aff_detection.streams.lang_enc=LANGENC \\ dataset_name=calvin_langDepthEndPt \\
aff_detection.model_cfg.freeze_encoder.lang=True \\
aff_detection.model_cfg.freeze_encoder.aff=False \\
wandb.logger.group=aff_exps
```
Available configurations for aff_detection:
- rn18_bert_pixel
- rn18_clip_pixel (rn18 visual encoder + clip language)
- rn50_bert_pixel
- rn50_clip_pixel (clip for visual and language)

Available language encoders:
- clip
- bert
- distilbert
- sbert

Script: [train_affordance.py](../hulc2/affordance/train_affordance.py)

Config: [train_affordance.yaml](../conf/affordance/train_affordance.yaml)

# Testing / Evaluation
### Running locally:
```
python -m hulc2.affordance.test_affordance
```
## Visualizing the predictions on the dataset
Script: [test_affordance.py](../hulc2/affordance/test_affordance.py)

Config: [test_affordance.yaml](../conf/affordance/test_affordance.yaml)

Usage:
```
python test_affordance.py checkpoint.train_folder=AFFORDANCE_TRAIN_FOLDER aff_detection.dataset.data_dir=DATASET_TO_TEST_PATH
```

## Testing move to a point given language annotation
Script: [test_move_to_pt.py](../hulc2/affordance/test_move_to_pt.py)

Config: [cfg_high_level.yaml](../conf/cfg_high_level.yaml)

## Evaluation model-based + model-free with affordance
Script: [evaluate_policy.py](../hulc2/evaluation/evaluate_policy.py)

### Running on a cluster:
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
