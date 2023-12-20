## Overview

This repo provides code and instructions for experimenting with the [PLEX architecture](https://microsoft.github.io/PLEX/) for robotic manipulation agents.


__Table of Contents__
- [Installation](#installation)
- [Meta-World experiments](#meta-world-experiments)
  * [Meta-World data setup](#meta-world-data-setup)
  * [Running the PLEX paper's Meta-World experiments](#running-the-plex-papers-meta-world-experiments)
- [Robosuite/Robomimic experiments](#robosuiterobomimic-experiments)
  * [Robosuite/Robomimic data setup](#robosuiterobomimic-data-setup)
  * [Running the PLEX paper's Robosuite/Robomimic experiments](#running-the-plex-papers-robosuiterobomimic-experiments)
- [Citing PLEX and the accompanying data](#citing-plex-and-the-accompanying-data)
- [Contributing](#contributing)
- [Trademarks](#trademarks)



## Installation

*NOTE:* this setup has been tested on Windows 11's WSL2 running Ubuntu 18.04 LTS and 20.04 LTS, as well as on native Ubuntu 18.04 LTS and 20.04 LTS.

Install [MuJoCo 2.1 binaries](https://github.com/openai/mujoco-py#install-mujoco). TLDR:

```
# Download and extract the binaries:
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.gz
mkdir ~/.mujoco
tar -xf mujoco.gz --directory ~/.mujoco
rm mujoco.gz

# Add these lines to your ~/.bashrc:
EXPORT LD_LIBRARY_PATH=$HOME/.mujoco/mujoco-210/bin:$LD_LIBRARY_PATH
EXPORT MJLIB_PATH=$HOME/.mujoco/mujoco-210/bin/libmujoco210.so

# Reload bashrc:
source ~/.bashrc
```

Create a virtual environment and install the PLEX repo there:

```
conda create -n plex pip=21.2.4 python=3.8
conda activate plex
conda install cudatoolkit=11.7 -c nvidia
git clone https://github.com/microsoft/PLEX.git
cd PLEX
pip install -e .
```

Create a directory for all the data you will use for training and models you will produce:
```
mkdir store
```

## Meta-World experiments

### Meta-World data setup

The [PLEX paper](https://arxiv.org/abs/2303.08789)'s Meta-World experiments use data that can be generated using scripted Meta-World policies with different levels of noise. These policies use a modified Meta-World flavor from the [PLEX-Metaworld](https://github.com/microsoft/PLEX-Metaworld) repo, which is installed automatically as one of PLEX's dependencies above.

To generate this training data, run the command below from the PLEX repo root. It will generate 100 trajectories for each of the 50 Meta-World tasks and extra 75 trajectories for 5 of these tasks, so it will take some time:

```
python scripts/gen_MW_data_for_PLEX.py --out=store/data/
```

The data will be placed in the `store` directory you created during PLEX setup.


### Running the PLEX paper's Meta-World experiments

#### First, pretrain EX.

To do so, run the following from the PLEX repo root:

```
python scripts/exps_on_MW.py --training_stage=ex --data_dir=store/data/ --log_dir=store/logs/
```

#### Next, complete PLEX pretraining by choosing a checkpoint with a pretrained EX and pretraining PL on top of it.

The checkpoints generated by the previous step's command contain pretrained EX versions, have file names starting with `pretr_EX__`, and can be found in a subdirectory of `<PLEX repo root>/store/logs` whose name is the timestamp when the command was run. Generally, choose the checkpoint from the latest iteration -- its name ends in `__latest.pt` As an example, the command below uses such a checkpoint from the log directory `store/logs/11-26-23_06.47.24_None`:
```
python scripts/exps_on_MW.py --training_stage=pl --data_dir=store/data/ --log_dir=store/logs --model_file=store/logs/11-26-23_06.47.24_None/pretr_EX__plK30_plL3_plH4_exK30_exL3_exH4_res84_bcTrue_la1_relposTrue__latest.pt
```

#### Finally, finetune a pretrained PLEX for a specific target task.

Pretrained PLEX checkpoints are contained in the `PLEX/store/logs` subdirectory generated by the previous step. Their file names start with `pretr_PLEX__`. As for the preceding step, usually the `...__latest.pt` is a good choice. To get the results in the PLEX paper, finetuning the chosen checkpoint needs to be repeated 10 times (i.e., for 10 seeds) for each of target tasks of Meta-World's ML50: `metaworld/hand-insert-v2/--TARGET_ROBOT--/noise0/`, `metaworld/door-lock-v2/--TARGET_ROBOT--/noise0/`, `metaworld/door-unlock-v2/--TARGET_ROBOT--/noise0/`, `metaworld/box-close-v2/--TARGET_ROBOT--/noise0/`, and `metaworld/bin-picking-v2/--TARGET_ROBOT--/noise0/`. To parallelize the evaluation rollouts, you can set the number of evaluation workers greater than 0. PLEX's results in the paper for each of the target tasks `T` were obtained by taking, for each seed, the maximum success rate `R` across all finetuning iterations and averaging `R` across 10 seeds.

Below is an example finetuning command that finetunes a pretrained PLEX from the previous step on the `hand-insert-v2/` task and runs evaluation episodes using 5 workers:

```
python scripts/exps_on_MW.py --training_stage=ft --data_dir=store/data/ --num_workers=5 --target_task=metaworld/hand-insert-v2/--TARGET_ROBOT--/noise0/  --log_dir=store/logs --model_file=store/logs/11-27-23_00.20.38_None/pretr_PLEX__plK30_plL3_plH4_exK30_exL3_exH4_res84_bcTrue_la1_relposTrue__latest.pt
```

**NOTE**: If running PLEX on CPU, set `--num_workers=0`. Running PLEX on CPU with `--num_workers` > 0 will throw a  `"To use CUDA with multiprocessing, you must use the 'spawn' start method"` error, and using the `spawn` method will throw another error.


## Robosuite/Robomimic experiments

### Robosuite/Robomimic data setup

The training demonstrations for PLEX's Robosuite/Robomimic experiments were partly collected by us and partly come from the Robomimic dataset. For Robosuite, we collected 75 trajectories for `Door`, `Stack`, `PickPlaceMilk`, `PickPlaceBread`, `PickPlaceCereal`, and `NutAssemblyRound` tasks. They are available from the [**Microsoft Download Center**](https://www.microsoft.com/en-us/download/details.aspx?id=105664).

For Robomimic, demonstration sets for `Lift`, `NutAssemblySquare` and `PickPlaceCan`, 200 trajectories per task, are availabe at the [**Robomimic webpage**](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html#downloading).

Both the Robosuite and Robomimic data we used is of "professional human (ph)" quality, and each experiment involved at most 75 demonstrations per task. Before runnning the experiments, the Robosuite and Robomimic data in the `raw` format needs to be downloaded and processed. The following command runs the download and processing end-to-end:

```
python scripts/setup_RR_data_for_PLEX.py --out=store/data
```

If everything worked, in `store/data` for each of the Robotsuite and Robomimic tasks you should see directory paths such as `robouite/Stack/Panda/raw/` and `Stack/Panda/ph/`, where `Stack/Panda/raw/` contains one or several hdf5 files with raw data and `Stack/Panda/ph/` contains the same number of much larger hdf5 files that contain the processed demonstrations.




### Running the PLEX paper's Robosuite/Robomimic experiments

The following command run from the PLEX repo root trains PLEX with the relative position encoding in BC mode on Robosuite's `Stack` task on 10 randomly chosen demonstration trajectories on 1 seed :
```
python scripts/exps_on_RR.py --arch=plex-rel --data_dir=store/data/ --log_dir=store/logs/ --target_task=robosuite/Stack/--TARGET_ROBOT--/ph/ --max_tt=10 --num_workers=5
```

To train PLEX for the same `Stack` target task and the same hyperparameters but with the absolute position encoding, run

```
python scripts/exps_on_RR.py --arch=plex-abs --data_dir=store/data/ --log_dir=store/logs/ --target_task=robosuite/Stack/--TARGET_ROBOT--/ph/ --max_tt=10 --num_workers=5
```

To train the Decision Transformer, which uses global position encodring but is otherwise comparable to PLEX in size and training hyperparameters, run

```
python scripts/exps_on_RR.py --arch=dt --data_dir=store/data/ --log_dir=store/logs/ --target_task=robosuite/Stack/--TARGET_ROBOT--/ph/ --max_tt=10 --num_workers=5
```

Generating the curves in Figure 3 of the [PLEX paper](https://arxiv.org/abs/2303.08789) requires running these commands for each of the tasks (using `robosuite/Door/--TARGET_ROBOT--/ph/`, `robosuite/Stack/--TARGET_ROBOT--/ph/`, `robosuite/PickPlaceMilk/--TARGET_ROBOT--/ph/`, `robosuite/PickPlaceBread/--TARGET_ROBOT--/ph/`, `robosuite/PickPlaceCereal/--TARGET_ROBOT--/ph/`, `robosuite/NutAssemblyRound/--TARGET_ROBOT--/ph/`, `robomimic/Lift/--TARGET_ROBOT--/ph/`, `robomimic/NutAssemblySquare/--TARGET_ROBOT--/ph/`, and `robomimic/PickPlaceCan/--TARGET_ROBOT--/ph/` as `--target_task`) for each `--max_tt` value in {5, 10, 25, 50, 75} for 10 seeds.


**NOTE**: As with Meta-World experiments, if running PLEX on CPU, set `--num_workers=0`. Running PLEX on CPU with `--num_workers` > 0 will throw a `"To use CUDA with multiprocessing, you must use the 'spawn' start method"` error, and using the `spawn` method will throw another error.


## Citing PLEX and the accompanying data

If you find PLEX, its implementation, or the accompanying Robosuite dataset useful in your work, please cite it as follows:

```
@inproceedings{thomas2023plex,
  title={PLEX: Making the Most of the Available Data for Robotic Manipulation Pretraining},
  author={Garrett Thomas and Ching-An Cheng and Ricky Loynd and Felipe Vieira Frujeri and Vibhav Vineet and Mihai Jalobeanu and Andrey Kolobov},
  booktitle={CoRL},
  year={2023}
  eprint={2303.08789},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
  url={https://arxiv.org/abs/2303.08789}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.