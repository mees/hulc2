# HULC++
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[<b>Grounding Language with Visual Affordances over Unstructured Data</b>](https://arxiv.org/pdf/2210.01911.pdf)

[Oier Mees](https://www.oiermees.com/), Jessica Borja-Diaz, [Wolfram Burgard](http://www2.informatik.uni-freiburg.de/~burgard)


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
conda create -n hulc2_venv python=3.8  # or use virtualenv
conda activate hulc2_venv
sh install.sh
```
If you encounter problems installing pyhash, you might have to downgrade setuptools to a version below 58.


## Acknowledgements

This work uses code from the following open-source projects and datasets:

#### CALVIN
Original:  [https://github.com/mees/calvin](https://github.com/mees/calvin)
License: [MIT](https://github.com/mees/calvin/blob/main/LICENSE)

#### Sentence-Transformers
Original:  [https://github.com/UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers)
License: [Apache 2.0](https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE)

## Citations

If you find the code useful, please cite:

**HULC++**
```bibtex
@article{mees22hulc2,
  title={Grounding  Language  with  Visual  Affordances  over  Unstructured  Data},
  author={Oier Mees and Jessica Borja-Diaz and Wolfram Burgard},
  journal={arXiv preprint arXiv:2210.01911},
  year={2022}
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