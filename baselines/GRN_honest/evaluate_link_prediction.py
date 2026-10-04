# -*- coding: utf-8 -*-
"""Evaluate a trained GRN checkpoint under a negative sampling strategy (random / historical / inductive), DyGLib's
evaluate_link_prediction.py flow with GRN's streaming state rebuilt from the training events (the released
repository has no evaluation script; its --negative_sample_strategy option was never used)."""
import logging
import time
import sys
import os
import json
import warnings

import numpy as np
import torch.nn as nn

from models.GRN import GRN
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction, replay_positive_events
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
from train_link_prediction import run_evaluations, average_metrics


if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    args = get_link_prediction_args(is_evaluation=True)
    assert args.model_name == 'GRN', 'this tree evaluates GRN only'

    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    if args.negative_sample_strategy != 'random':
        val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids,
                                                   interact_times=full_data.node_interact_times, last_observed_time=train_data.node_interact_times[-1],
                                                   negative_sample_strategy=args.negative_sample_strategy, seed=0)
        new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids,
                                                            interact_times=new_node_val_data.node_interact_times, last_observed_time=train_data.node_interact_times[-1],
                                                            negative_sample_strategy=args.negative_sample_strategy, seed=1)
        test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids,
                                                    interact_times=full_data.node_interact_times, last_observed_time=val_data.node_interact_times[-1],
                                                    negative_sample_strategy=args.negative_sample_strategy, seed=2)
        new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids,
                                                             interact_times=new_node_test_data.node_interact_times, last_observed_time=val_data.node_interact_times[-1],
                                                             negative_sample_strategy=args.negative_sample_strategy, seed=3)
    else:
        val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
        new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
        test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
        new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    loss_func = nn.BCELoss()
    ctx = {'full_neighbor_sampler': full_neighbor_sampler, 'loss_func': loss_func,
           'val_loader': val_idx_data_loader, 'val_sampler': val_neg_edge_sampler, 'val_data': val_data,
           'new_node_val_loader': new_node_val_idx_data_loader, 'new_node_val_sampler': new_node_val_neg_edge_sampler, 'new_node_val_data': new_node_val_data,
           'test_loader': test_idx_data_loader, 'test_sampler': test_neg_edge_sampler, 'test_data': test_data,
           'new_node_test_loader': new_node_test_idx_data_loader, 'new_node_test_sampler': new_node_test_neg_edge_sampler, 'new_node_test_data': new_node_test_data}

    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

    for run in range(args.seed_start, args.seed_start + args.num_runs):

        set_random_seed(seed=run)
        args.seed = run
        args.load_model_name = f'{args.model_name}_seed{args.seed}'
        args.save_result_name = f'{args.negative_sample_strategy}_negative_sampling_{args.model_name}_seed{args.seed}'

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/", exist_ok=True)
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")
        logger.info(f'configuration is {args}')

        dynamic_backbone = GRN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                               time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                               num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                               max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                    hidden_dim=node_raw_features.shape[1], output_dim=1)
        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B')

        load_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.load_model_name}"
        early_stopping = EarlyStopping(patience=0, save_model_folder=load_model_folder,
                                       save_model_name=args.load_model_name, logger=logger, model_name=args.model_name)
        early_stopping.load_checkpoint(model, map_location='cpu')
        model = convert_to_gpu(model, device=args.device)

        logger.info(f'get final performance on dataset {args.dataset_name}...')
        model[0].set_neighbor_sampler(train_neighbor_sampler)
        model[0].reset_state()
        replay_positive_events(model, train_idx_data_loader, train_data)
        train_backup = model[0].backup_state()
        ev = run_evaluations(model, train_backup, ctx, with_test=True)
        val_losses, val_metrics = ev['val']
        new_node_val_losses, new_node_val_metrics = ev['new_node_val']
        test_losses, test_metrics = ev['test']
        new_node_test_losses, new_node_test_metrics = ev['new_node_test']

        val_metric_dict = average_metrics(val_metrics)
        new_node_val_metric_dict = average_metrics(new_node_val_metrics)
        test_metric_dict = average_metrics(test_metrics)
        new_node_test_metric_dict = average_metrics(new_node_test_metrics)

        logger.info(f'validate loss: {np.mean(val_losses):.4f}')
        for metric_name, value in val_metric_dict.items():
            logger.info(f'validate {metric_name}, {value:.4f}')
        logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
        for metric_name, value in new_node_val_metric_dict.items():
            logger.info(f'new node validate {metric_name}, {value:.4f}')
        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name, value in test_metric_dict.items():
            logger.info(f'test {metric_name}, {value:.4f}')
        logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
        for metric_name, value in new_node_test_metric_dict.items():
            logger.info(f'new node test {metric_name}, {value:.4f}')

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        val_metric_all_runs.append(val_metric_dict)
        new_node_val_metric_all_runs.append(new_node_val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        if run < args.seed_start + args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        result_json = {
            "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
            "new node validate metrics": {metric_name: f'{new_node_val_metric_dict[metric_name]:.4f}' for metric_name in new_node_val_metric_dict},
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
            "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
        }
        result_json = json.dumps(result_json, indent=4)
        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        logger.info(f'save negative sampling results at {save_result_path}')

    logger.info(f'metrics over {args.num_runs} runs:')
    for name, runs in [('validate', val_metric_all_runs), ('new node validate', new_node_val_metric_all_runs),
                       ('test', test_metric_all_runs), ('new node test', new_node_test_metric_all_runs)]:
        for metric_name in runs[0].keys():
            values = [r[metric_name] for r in runs]
            logger.info(f'{name} {metric_name}, {values}')
            logger.info(f'average {name} {metric_name}, {np.mean(values):.4f} '
                        f'± {(np.std(values, ddof=1) if len(values) > 1 else 0.0):.4f}')

    sys.exit()
