#!/bin/bash
echo "Running from src: $1"
echo "Output to: $2"

# Job to perform
# source ~/.bashrc
# conda activate $1
# srun python ${@:2}

python preprocess_real_data.py --dataset_root $1 --output_dir $2_processed
python render_low_freq.py --dataset_root $2_processed --output_dir  $2_15hz
python split_dataset.py --dataset_root $2_15hz
