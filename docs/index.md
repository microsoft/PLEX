---
title: PLEX: 'Making the Most of the Available Data for Robotic Manipulation Pretraining' layout: default
---

## Motivation
Data-driven methods for producing robotic manipulation policies have the potential to be effective and scalable, replacing manually engineered controllers. At the same time, recent trends in natural language processing have shown that transformers trained on large amounts of data can exhibit impressive capabilities. We aim to combine these threads and work towards a scalable transformer-based model for robotic manipulation that can take advantage of all available, relevant forms of data.

In particular, we consider three common classes of data available for training robotic manipulators:

 * **Multi-task video demonstrations (MTVD)**, which contain high-quality and potentially annotated demonstrations for a variety of tasks, but have no explicit action information for a robot to mimic

 * **Visuomotor trajectories (VMT)**, which consist of paired sequences of observations and robots’ actions, but do not necessarily correspond to meaningful tasks

 * **Target-task demonstrations (TTD)**, which are high-quality trajectories for a specific task of interest, collected using the robot of interest

Note that the more widely available data tends to be less informative (e.g. videos), while the most informative data (e.g. target demonstrations) is the scarcest.

## Method
Our ***PL*anning-*EX*ecution (PLEX)** architecture separates the model into a *planner*, which predicts the future observations (in a latent space), and an *executor*, which predicts the actions needed to effect those future observations:
![architecture](./assets/img/PLEX.png)
During pretraining, the planner is trained using MTVD data, and the executor is trained using VMT data.

During fine-tuning, the entire network is trained end-to-end using TTD data.


## Experiments

![Examples of tasks used in experiments](./assets/img/robots.png)

We investigate the performance of PLEX on the Meta-World benchmark by training PLEX on videos from 45 pretraining tasks of Meta-World's ML50 split and finetuning it on each of ML50's 5 other tasks:
![Meta-World results](./assets/img/metaworld-results.png)
Interestingly, PLEX works fairly well even zero-shot, and needs only videos of the target task demonstrations.

To assess the effectiveness of PLEX's relative position encoding, we also evaluate PLEX on the Robosuite/Robomimic benchmark. The results demonstrate that the version with relative position encoding outperforms PLEX with absolute encoding -- the typical position encoding choice in the transformer literature -- as well as the vanilla Decision Transformer, which relies on the global position encoding scheme:
![Robosuite results](./assets/img/robosuite-results.png)

**NOTE on the Robosuite dataset:** For the Robosuite experiments, we gathered a dataset of high-qualty demonstration trajectories for Robosuite's `Door`, `Stack`, `PickPlaceMilk`, `PickPlaceBread`, `PickPlaceCereal`, and `NutAssemblyRound` tasks, 75 demonstrations per each. The dataset is available from the [**Microsoft Download Center**](https://www.microsoft.com/en-us/download/details.aspx?id=105664), and instructions for processing it can be found [here](https://github.com/microsoft/PLEX#robosuiterobomimic-data-setup).