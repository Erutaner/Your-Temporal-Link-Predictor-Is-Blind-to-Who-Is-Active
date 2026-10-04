> This copy carries the corrections described in `../README.md`; the node-classification script mentioned below is not
> included.  The rest of this file is the upstream README.

# Graph Retention Networks  

![License](https://img.shields.io/badge/License-MIT-blue)

This repository provides the implementation code for the paper **Graph Retention Networks for Dynamic Graphs**. 

## Introduction
The proposed Graph Retention Network (GRN) incorporates the concept of retention into dynamic graph data, introducing three key computational paradigms: **parallelizable training**, **O(1) low-cost inference**, and **long-term batch training**. This architecture achieves an optimal balance among **effectiveness**, **efficiency**, and **scalability**.

The GRN is designed as a unified architecture that does not rely on sampling or truncation neighbor aggregation strategies, does not require separate memory modules, and is more efficient than attention-based mechanisms.

<img src="https://anonymous.4open.science/api/repo/GraphRetentionNet/file/figures/architecture.png?v=bf5c8c23" width="800" />


## Benchmark Datasets

The datasets used in this project are available [here](https://zenodo.org/records/7213796#.Y1cO6y8r30o). Before training the model, we need to download the datasets and unzip them into the `./processed_data` directory.

## Dependencies
All required dependencies are listed in the `requirements.txt` file.

## Hyperparameter Configurations
The optimal parameter configurations for both the link prediction and node classification tasks are saved in `utils/load_configs.py`.

## Usage

### Dynamic Link Prediction Task
To train the *GRN* model on the *wikipedia* dataset using GPU for 5 runs, we can execute:

```{bash}
python train_link_prediction.py --dataset_name wikipedia --model_name GRN --num_runs 5 --gpu 0
```

To train GRN on wikipedia with the best configurations:

```{bash}
python train_link_prediction.py --dataset_name wikipedia --model_name GRN --load_best_configs --num_runs 5 --gpu 0
```

Three negative sampling strategies are available in the script, i.e. `random`, `historical`, and `inductive`. To train and evaluate the GRN model using the `inductive` sampling strategy:

```{bash}
python train_link_prediction.py --dataset_name wikipedia --model_name GRN --negative_sample_strategy inductive --load_best_configs --num_runs 5 --gpu 0
```

### Dynamic Node Classfication Task

To train the model for the dynamic node classification task:

```{bash}
python train_node_classification.py --dataset_name wikipedia --model_name GRN --num_runs 5 --gpu 0
```

To train *GRN* on *wikipedia* dataset using the best configurations for the dynamic node classification task:

```{bash}
python train_node_classification.py --dataset_name wikipedia --model_name GRN --load_best_configs --num_runs 5 --gpu 0
```
