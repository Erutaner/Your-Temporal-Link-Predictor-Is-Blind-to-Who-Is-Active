# -*- coding: utf-8 -*-
"""GRN link-prediction training with a strict evaluation protocol (see models/GRN.py).

Differences from the released train_link_prediction.py (GRN branch):
  * the positive pair of a batch is (src, dst); the released code computed the
    "positive" embeddings for (src, negative dst) and the negatives for the same
    pair after the state had absorbed it, so the true destinations were never
    trained on;
  * the state is backed up / restored by value, and the final evaluation
    rebuilds the state by streaming the training events through the selected
    checkpoint (the released final evaluation ran on a state that had already
    consumed the validation, and on test epochs the test, events);
  * DyGLib's data split (the released loader removed the edges of the held-out
    nodes from the transductive validation / test sets);
  * seeds seed_start .. seed_start + num_runs - 1; no efficiency probes.
"""
import logging
import time
import os
import json
import shutil
import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.GRN import GRN
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction, replay_positive_events
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args


def average_metrics(metrics):
    return {name: float(np.mean([m[name] for m in metrics])) for name in metrics[0].keys()}


def run_evaluations(model, train_backup, ctx, with_test):
    """Released state flow: validation from the post-train state, new-node validation from the post-train state,
    test and new-node test from the post-validation state (each restored by value)."""
    grn = model[0]
    grn.reset_state(train_backup)
    out = {}
    out['val'] = evaluate_model_link_prediction(model_name='GRN', model=model, neighbor_sampler=ctx['full_neighbor_sampler'],
                                                evaluate_idx_data_loader=ctx['val_loader'], evaluate_neg_edge_sampler=ctx['val_sampler'],
                                                evaluate_data=ctx['val_data'], loss_func=ctx['loss_func'])
    val_backup = grn.backup_state()
    grn.reset_state(train_backup)
    out['new_node_val'] = evaluate_model_link_prediction(model_name='GRN', model=model, neighbor_sampler=ctx['full_neighbor_sampler'],
                                                         evaluate_idx_data_loader=ctx['new_node_val_loader'], evaluate_neg_edge_sampler=ctx['new_node_val_sampler'],
                                                         evaluate_data=ctx['new_node_val_data'], loss_func=ctx['loss_func'])
    if with_test:
        grn.reset_state(val_backup)
        out['test'] = evaluate_model_link_prediction(model_name='GRN', model=model, neighbor_sampler=ctx['full_neighbor_sampler'],
                                                     evaluate_idx_data_loader=ctx['test_loader'], evaluate_neg_edge_sampler=ctx['test_sampler'],
                                                     evaluate_data=ctx['test_data'], loss_func=ctx['loss_func'])
        grn.reset_state(val_backup)
        out['new_node_test'] = evaluate_model_link_prediction(model_name='GRN', model=model, neighbor_sampler=ctx['full_neighbor_sampler'],
                                                              evaluate_idx_data_loader=ctx['new_node_test_loader'], evaluate_neg_edge_sampler=ctx['new_node_test_sampler'],
                                                              evaluate_data=ctx['new_node_test_data'], loss_func=ctx['loss_func'])
    return out


def main():
    warnings.filterwarnings('ignore')
    args = get_link_prediction_args(is_evaluation=True)
    assert args.model_name == 'GRN', 'this tree trains GRN only'

    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)
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
        args.save_model_name = f'{args.model_name}_seed{args.seed}'

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
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
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)
        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        for epoch in range(args.num_epochs):

            model.train()
            model[0].set_neighbor_sampler(train_neighbor_sampler)
            model[0].reset_state()

            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      batch_edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      batch_edge_ids=batch_edge_ids,
                                                                      edges_are_positive=False)

                positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

                predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                loss = loss_func(input=predicts, target=labels)
                train_losses.append(loss.item())
                train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=2)
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

            train_backup = model[0].backup_state()
            ev = run_evaluations(model, train_backup, ctx, with_test=((epoch + 1) % args.test_interval_epochs == 0))
            val_losses, val_metrics = ev['val']
            new_node_val_losses, new_node_val_metrics = ev['new_node_val']

            logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
            for metric_name, value in average_metrics(train_metrics).items():
                logger.info(f'train {metric_name}, {value:.4f}')
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name, value in average_metrics(val_metrics).items():
                logger.info(f'validate {metric_name}, {value:.4f}')
            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            for metric_name, value in average_metrics(new_node_val_metrics).items():
                logger.info(f'new node validate {metric_name}, {value:.4f}')
            if 'test' in ev:
                test_losses, test_metrics = ev['test']
                new_node_test_losses, new_node_test_metrics = ev['new_node_test']
                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                for metric_name, value in average_metrics(test_metrics).items():
                    logger.info(f'test {metric_name}, {value:.4f}')
                logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
                for metric_name, value in average_metrics(new_node_test_metrics).items():
                    logger.info(f'new node test {metric_name}, {value:.4f}')

            # select the best model based on all the validate metrics
            val_metric_indicator = [(metric_name, value, True) for metric_name, value in average_metrics(val_metrics).items()]
            early_stop = early_stopping.step(val_metric_indicator, model)
            if early_stop:
                break

        # load the best model and rebuild its streaming state from the training events
        early_stopping.load_checkpoint(model)
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
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)

    logger.info(f'metrics over {args.num_runs} runs:')
    for name, runs in [('validate', val_metric_all_runs), ('new node validate', new_node_val_metric_all_runs),
                       ('test', test_metric_all_runs), ('new node test', new_node_test_metric_all_runs)]:
        for metric_name in runs[0].keys():
            values = [r[metric_name] for r in runs]
            logger.info(f'{name} {metric_name}, {values}')
            logger.info(f'average {name} {metric_name}, {np.mean(values):.4f} '
                        f'± {(np.std(values, ddof=1) if len(values) > 1 else 0.0):.4f}')
    logger.removeHandler(fh)
    logger.removeHandler(ch)


if __name__ == '__main__':
    main()
