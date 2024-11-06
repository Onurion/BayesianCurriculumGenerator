# Bayesian Curriculum Generation Algorithm

## Overview
This repository hosts the AeroRival Pursuit implementation of the Bayesian Curriculum Generation Algorithm (BCG), designed for curriculum learning in sparse reward reinforcement learning contexts. For access to the MiniGrid environment implementation, please switch to the `main` branch. This dynamic and adaptive approach is showcased in various settings, demonstrating enhanced learning efficiency.

## Key Features
- **Dynamic Task Generation:** Uses Bayesian networks for real-time task creation, adjusting complexity based on learning progression.
- **Unsupervised Task Classification:** Implements clustering for task categorization based on difficulty, applicable to image outputs and scalar values without extensive task-specific training.
- **Adaptive Learning:** Modulates task difficulty in response to agent performance, enhancing learning efficiency by presenting progressively challenging tasks or adjusting current task complexity.

## Environments
The environments for training both the proposed approach and benchmark algorithms are illustrated below:

![Training Environments](figs/Aerorival_tasks.png)

## Results
The results from the test evaluations are presented here:

![Test Results](figs/Results_aerorival_all_tasks.png)

## Installation and Training
To install the environment and execute the training, run the following commands:
```bash
git clone -b aerorival https://github.com/Onurion/BayesianCurriculumGenerator
cd BayesianCurriculumGenerator
pip install -r requirements.txt 
python train_aerorival.py --train_type "Curriculum"
```

## Paper Review

This paper is currently under review in Engineering Science and Technology, an International Journal journal.

