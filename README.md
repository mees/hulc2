# HULC++
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[<b>Grounding Language with Visual Affordances over Unstructured Data</b>](https://arxiv.org/pdf/2210.01911.pdf)

[Oier Mees](https://www.oiermees.com/), Jessica Borja-Diaz, [Wolfram Burgard](http://www2.informatik.uni-freiburg.de/~burgard)

#### HULC++ was nominated as finalist for the Best Paper Award in Robot Learning at the IEEE International Conference on Robotics and Automation (ICRA) 2023 


We present **HULC++** (**H**ierarchical **U**niversal **L**anguage **C**onditioned Policies) 2.0, a novel approach to efficiently  learn general-purpose language-conditioned robot skills from
unstructured, offline and reset-free data by exploiting a self-supervised visuo-lingual affordance model,  which requires annotating as little as 1% of the total data with language.
We find that when paired with LLMs to break down abstract natural language instructions into subgoals via few-shot prompting, our method
is capable of completing long-horizon, multi-tier tasks in the real world, while *requiring an order of magnitude less data* than previous approaches.

![](media/hulc2.gif)

## Installation
To begin, clone this repository locally
```bash
git clone --recurse-submodules https://github.com/mees/hulc2.git
export HULC_ROOT=$(pwd)/hulc2

```
Install requirements:
```bash
cd $HULC_ROOT
conda create -n hulc2_venv python=3.9  # or use virtualenv
conda activate hulc2_venv
sh install.sh
```
If you encounter problems installing pyhash, you might have to downgrade setuptools to a version below 58.

## Download
### Task-Agnostic Real World Robot Play Dataset
We host the multimodal 9 hours of human teleoperated [play dataset on kaggle](https://www.kaggle.com/datasets/oiermees/taco-robot).
Download the dataset:
```
cd $HULC2_ROOT/dataset
kaggle datasets download -d oiermees/taco-robot
```
### CALVIN Dataset
If you want to train on the simulated [CALVIN](https://github.com/mees/calvin) dataset, choose a split with:
```bash
cd $HULC2_ROOT/dataset
sh download_data.sh D | ABC | ABCD | debug
```
If you want to get started without downloading the whole dataset, use the argument `debug` to download a small debug dataset (1.3 GB).

### Pre-trained Models
Download the [HULC2 model checkpoints](http://hulc2.cs.uni-freiburg.de/model_weights/real_world_checkpoints.zip) trained on the [real world play dataset](https://www.kaggle.com/datasets/oiermees/taco-robot).
```
$ wget http://hulc2.cs.uni-freiburg.de/model_weights/real_world_checkpoints.zip
$ unzip real_world_checkpoints.zip
```
## Training
**1.** [Affordance Model](./docs/affordance_condensed.md)

**2.** Model-free Policy
By default a training for the CALVIN simulated environment will be started:
```
python hulc2/training.py trainer.devices=-1 datamodule.root_data_dir=path/to/dataset
```
To train the policy on the real world dataset, you can change the default config file:
```
python hulc2/training.py trainer.devices=-1 datamodule.root_data_dir=path/to/dataset --config-name cfg_low_level_rw
```
By default we backprop through the language encoder, if you want to use pretrained language embeddings, you can add the following arguments `datamodule.datasets.lang_dataset.load_lang_embeddings=true
model/language_encoder=none`


## Evaluation
See detailed inference instructions on the [CALVIN repo](https://github.com/mees/calvin#muscle-evaluation-the-calvin-challenge).
```
python hulc++/evaluation/evaluate_policy.py --dataset_path <PATH/TO/DATASET> --train_folder <PATH/TO/TRAINING/FOLDER>
```
Set `--train_folder $HULC2_ROOT/checkpoints/HULC2_D_D` to evaluate our [pre-trained models](#pre-trained-models).

Optional arguments:

- `--checkpoint <PATH/TO/CHECKPOINT>`: by default, the evaluation loads the last checkpoint in the training log directory.
You can instead specify the path to another checkpoint by adding this to the evaluation command.
- `--debug`: print debug information and visualize environment.

## Acknowledgements

This work uses code from the following open-source projects and datasets:

#### CALVIN
Original:  [https://github.com/mees/calvin](https://github.com/mees/calvin)
License: [MIT](https://github.com/mees/calvin/blob/main/LICENSE)

#### HULC
Original:  [https://github.com/mees/hulc](https://github.com/mees/hulc)
License: [MIT](https://github.com/mees/hulc/blob/main/LICENSE)

#### VAPO
Original:  [https://github.com/mees/vapo](https://github.com/mees/vapo)
License: [MIT](https://github.com/mees/vapo/blob/main/LICENSE)

#### Robot-IO
Original:  [https://github.com/mees/robot_io](https://github.com/mees/robot_io)
License: [LGPL-3](https://github.com/mees/robot_io/blob/main/LICENSE)

#### Sentence-Transformers
Original:  [https://github.com/UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers)
License: [Apache 2.0](https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE)

#### CLIPort
Original:  [https://github.com/cliport/cliport](https://github.com/cliport/cliport)
License: [Apache 2.0](https://github.com/cliport/cliport/blob/master/LICENSE)

#### R3M
Original:  [https://github.com/facebookresearch/r3m](https://github.com/facebookresearch/r3m)
License: [MIT](https://github.com/facebookresearch/r3m/blob/main/LICENSE)

#### CLIP
Original: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)
License: [MIT](https://github.com/openai/CLIP/blob/main/LICENSE)

## Citations

If you find the code useful, please consider citing:

**HULC++**
```bibtex
@inproceedings{mees23hulc2,
title={Grounding  Language  with  Visual  Affordances  over  Unstructured  Data},
author={Oier Mees and Jessica Borja-Diaz and Wolfram Burgard},
booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
year={2023},
address = {London, UK}
}
```

**HULC**
```bibtex
@article{mees2022hulc,
  author={Oier Mees and Lukas Hermann and Wolfram Burgard},
  title={What Matters in Language Conditioned Robotic Imitation Learning Over Unstructured Data},
  journal={IEEE Robotics and Automation Letters (RA-L)},
  volume={7},
  number={4},
  pages={11205-11212},
  year={2022}
}
```

**VAPO**
```bibtex
@inproceedings{borja22icra,
author = {Jessica Borja-Diaz and Oier Mees and Gabriel Kalweit and  Lukas Hermann and Joschka Boedecker and Wolfram Burgard},
title = {Affordance Learning from Play for Sample-Efficient Policy Learning},
booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation  (ICRA)},
year = 2022,
address = {Philadelphia, USA}
}
```

**CALVIN**
```bibtex
@article{mees2022calvin,
author = {Oier Mees and Lukas Hermann and Erick Rosete-Beas and Wolfram Burgard},
title = {CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks},
journal={IEEE Robotics and Automation Letters (RA-L)},
volume={7},
number={3},
pages={7327-7334},
year={2022}
}
```

## License

MIT License
