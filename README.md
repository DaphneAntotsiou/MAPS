# Modular Adaptive Policy Selection for Multi-Task Imitation Learning through Task Division
Python 3.x implementation of the MAPS system presented in the ICRA 2022 paper [Modular Adaptive Policy Selection for Multi-Task Imitation Learning through Task Division](https://arxiv.org/abs/2203.14855).

## Requirements
* python 3.6
* mujoco-py 1.50
* gym 0.12
* tensorflow 1.12
* [baselines](https://github.com/openai/baselines)
* [gym-extensions](https://github.com/Breakend/gym-extensions)
* [metaworld](https://github.com/rlworkgroup/metaworld)

## Installation
To install the maps package, open a terminal, go in the MAPS directory, and run
```
pip install -e .
```

## Training and evaluation instructions
The MT10 dataset is located in maps/data/tasks_10_traj_20_onehot.npz. The rest of the datasets can be obtained by training a separate TRPO agent for each of the environments of [gym-extensions](https://github.com/Breakend/gym-extensions).

### Train
To train a MAPS model, use the maps/train.py script with all the necessary parameters.

For example, to run the MT10 set with 10 modules, 15 trajectories, and specific parameters, go to the maps directory and run
```
python3 train.py --dataset data/tasks_10_traj_20_onehot.npz --module_num 10 --traj_limitation 15 --sharing 1 --exploration 0.1 --sparsity 0.5 --smooth 1
```
This will train a model and then run a multi-threaded evaluation. 

### Evaluate performance
To run an evaluation on a model, use the maps/multi_eval.py script.

For example, to run evaluation on an entire MT10 run in directory maps/checkpoint/tasks_10_traj_20_short_onehot-10_15_3layer_32bs_1.0_0.1_0.5_1.0_0.75/1 go to the maps directory and run
```
python3 multi_eval.py --dataset data/tasks_10_traj_20_onehot.npz --load_model_path checkpoint/tasks_10_traj_20_short_onehot-10_15_3layer_32bs_1.0_0.1_0.5_1.0_0.75/1 --eval_hours
```

### Evaluate module distribution
To get a module distribution of a specific model, use the multi_eval.py script and disable the eval_hours flag.

For example, to evaluate, get the model distribution, and render the result of task 'push' of the MT10 model at maps/checkpoint/tasks_10_traj_20_short_onehot-10_15_3layer_32bs_1.0_0.1_0.5_1.0_0.75/1/30000, go to the maps directory and run
```
python3 multi_eval.py --dataset data/tasks_10_traj_20_onehot.npz --env_id push-v2 --load_model_path checkpoint/tasks_10_traj_20_short_onehot-10_15_3layer_32bs_1.0_0.1_0.5_1.0_0.75/1/30000 --no-eval_hours --render
```

### View a specific module
To render the result of a specific module for a particular run, use the evaluate_module.py script.

For example, to render the result of task 'push' using only module 6, go to the maps directory and run
```
python3 evaluate_module.py --dataset data/tasks_10_traj_20_onehot.npz --env_id push-v2 --module 5 --load_model_path checkpoint/tasks_10_traj_20_short_onehot-10_15_3layer_32bs_1.0_0.1_0.5_1.0_0.75/1/30000 --render
```
Note: the modules in evaluate_module.py are zero-indexed.

## MT10 Dataset
The dataset for the MT10 task set can be found in the "maps/data" directory. It contains 20 trajectories performed by a human expert for each of the 10 tasks. The task ids for the trajectories are presented in a one-hot vector.
