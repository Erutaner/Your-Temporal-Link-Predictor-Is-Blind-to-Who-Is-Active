import logging
import time
import sys
import os
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"

import numpy as np
import warnings
import json
import torch.nn as nn
from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.DSRD import DSRD
from models.TPNet import RandomProjectionModule, TPNet
from models.NAT import NAT
from models.modules import LinkPredictor_v1, LinkPredictor_v2
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction, evaluate_edge_bank_link_prediction
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
from utils.utils import set_thread

def evaluate_link_prediction(args=None):

    warnings.filterwarnings('ignore')

    if args is None:
        # get arguments
        args = get_link_prediction_args(is_evaluation=True)
        
    args.seed = 0
    # set up logger
    args.load_model_name = f'{args.model_name}_seed{args.seed}'
    args.save_result_name = f'{args.negative_sample_strategy}_negative_sampling_{args.model_name}_seed{args.seed}'

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/", exist_ok=True)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log")
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # get data for training, validation and testing
    # PINT and TPNet require tuning of hyperparameters related to timestamps (i.e., args.pint_beta and args.time_decay_weight).
    # for non-discrete-time graphs, the time granularity is already in seconds.
    # to ensure consistent hyperparameter search range across all datasets, the time granularity of discrete-time graphs is also converted to seconds.
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
                                 logger=logger, convert_time=(args.use_random_projection or args.model_name == 'PINT'))

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    if args.negative_sample_strategy != 'random':
        val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids,
                                                   dst_node_ids=full_data.dst_node_ids,
                                                   interact_times=full_data.node_interact_times,
                                                   last_observed_time=train_data.node_interact_times[-1],
                                                   negative_sample_strategy=args.negative_sample_strategy, seed=0)
        new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids,
                                                            dst_node_ids=new_node_val_data.dst_node_ids,
                                                            interact_times=new_node_val_data.node_interact_times,
                                                            last_observed_time=train_data.node_interact_times[-1],
                                                            negative_sample_strategy=args.negative_sample_strategy,
                                                            seed=1)
        test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids,
                                                    dst_node_ids=full_data.dst_node_ids,
                                                    interact_times=full_data.node_interact_times,
                                                    last_observed_time=val_data.node_interact_times[-1],
                                                    negative_sample_strategy=args.negative_sample_strategy, seed=2)
        new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids,
                                                             dst_node_ids=new_node_test_data.dst_node_ids,
                                                             interact_times=new_node_test_data.node_interact_times,
                                                             last_observed_time=val_data.node_interact_times[-1],
                                                             negative_sample_strategy=args.negative_sample_strategy,
                                                             seed=3)
    else:
        val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids,
                                                   dst_node_ids=full_data.dst_node_ids, seed=0)
        new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids,
                                                            dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
        test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids,
                                                    dst_node_ids=full_data.dst_node_ids, seed=2)
        new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids,
                                                             dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))),
                                                batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))),
                                              batch_size=args.batch_size, shuffle=False)
    new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))),
                                                       batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))),
                                               batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))),
                                                        batch_size=args.batch_size, shuffle=False)

    # we separately evaluate EdgeBank, since EdgeBank does not contain any trainable parameters and has a different evaluation pipeline
    if args.model_name == 'EdgeBank':
        evaluate_edge_bank_link_prediction(args=args, train_data=train_data, val_data=val_data,
                                           test_idx_data_loader=test_idx_data_loader,
                                           test_neg_edge_sampler=test_neg_edge_sampler, test_data=test_data)
    else:
        val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

        for run in range(args.seed_start, args.seed_start + args.num_runs):

            # set the seed to ensure the results can be reproduced. For reproducing NAT and TPNet, the deterministic_alg needs to be set to True
            set_random_seed(seed=run, deterministic_alg=args.use_random_projection or args.model_name == 'NAT')


            args.seed = run
            
            args.load_model_name = f'{args.model_name}_seed{args.seed}'
            args.save_result_name = f'{args.negative_sample_strategy}_negative_sampling_{args.model_name}_seed{args.seed}'
            
            run_start_time = time.time()
            logger.info(f"********** Run {run + 1} starts. **********")

            logger.info(f'configuration is {args}')

            # create model
            random_projections = None
            if args.use_random_projection:
                # create the model to maintain the temporal walk matrices
                random_projections = RandomProjectionModule(node_num=node_raw_features.shape[0],
                                                            edge_num=edge_raw_features.shape[0],
                                                            dim_factor=args.rp_dim_factor,
                                                            num_layer=args.rp_num_layer,
                                                            time_decay_weight=args.rp_time_decay_weight,
                                                            device=args.device,
                                                            use_matrix=args.rp_use_matrix,
                                                            beginning_time=train_data.node_interact_times[0],
                                                            not_scale=args.rp_not_scale,
                                                            enforce_dim=args.enforce_dim)

            if args.model_name == 'TGAT':
                dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                        neighbor_sampler=full_neighbor_sampler,
                                        time_feat_dim=args.time_feat_dim, num_layers=args.num_layers,
                                        num_heads=args.num_heads, dropout=args.dropout, device=args.device)
            elif args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
                # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
                src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                    compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids,
                                                     train_data.node_interact_times)
                dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                               neighbor_sampler=full_neighbor_sampler,
                                               time_feat_dim=args.time_feat_dim, model_name=args.model_name,
                                               num_layers=args.num_layers, num_heads=args.num_heads,
                                               dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift,
                                               src_node_std_time_shift=src_node_std_time_shift,
                                               dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst,
                                               dst_node_std_time_shift=dst_node_std_time_shift, device=args.device,
                                               beta=args.pint_beta, num_hop=args.pint_hop)
            elif args.model_name == 'TPNet':
                dynamic_backbone = TPNet(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                         neighbor_sampler=full_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim,
                                         random_projections=None if args.encode_not_rp else random_projections,
                                         num_neighbors=args.num_neighbors,
                                         num_layers=args.num_layers,
                                         dropout=args.dropout,
                                         device=args.device)
            elif args.model_name == 'CAWN':
                dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                        neighbor_sampler=full_neighbor_sampler,
                                        time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim,
                                        walk_length=args.walk_length,
                                        num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
            elif args.model_name == 'TCL':
                dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                       neighbor_sampler=full_neighbor_sampler,
                                       time_feat_dim=args.time_feat_dim, num_layers=args.num_layers,
                                       num_heads=args.num_heads,
                                       num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
            elif args.model_name == 'GraphMixer':
                dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                              neighbor_sampler=full_neighbor_sampler,
                                              time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors,
                                              num_layers=args.num_layers, dropout=args.dropout, device=args.device)
            elif args.model_name == 'DyGFormer':
                dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                             neighbor_sampler=full_neighbor_sampler,
                                             time_feat_dim=args.time_feat_dim,
                                             channel_embedding_dim=args.channel_embedding_dim,
                                             patch_size=args.patch_size,
                                             num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                             max_input_sequence_length=args.max_input_sequence_length,
                                             device=args.device)
            elif args.model_name == 'DSRD':
                dynamic_backbone = DSRD(
                    node_raw_features=node_raw_features,
                    edge_raw_features=edge_raw_features,
                    neighbor_sampler=full_neighbor_sampler,
                    channel_embedding_dim=args.channel_embedding_dim,
                    time_feat_dim=args.time_feat_dim,
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                    dropout=args.dropout,
                    num_neighbors=args.num_neighbors,
                    device=args.device,
                )

            elif args.model_name == 'NAT':
                dynamic_backbone = NAT(n_feat=node_raw_features, e_feat=edge_raw_features, time_dim=args.time_feat_dim,
                                       num_neighbors=[1] + args.nat_num_neighbors, dropout=args.dropout,
                                       n_hops=args.num_layers,
                                       ngh_dim=args.nat_ngh_dim, device=args.device)
                dynamic_backbone.set_seed(args.seed)

            else:
                raise ValueError(f"Wrong value for model_name {args.model_name}!")

            # create the link prediction, NAT computes the edge embedding directly rather than node embedding, so it needs a different link predictor
            if args.model_name == 'NAT':
                link_predictor = LinkPredictor_v2(input_dim=node_raw_features.shape[1] + dynamic_backbone.self_dim * 2,
                                                  hidden_dim=node_raw_features.shape[1] + dynamic_backbone.self_dim * 2,
                                                  output_dim=1)
            else:
                link_predictor = LinkPredictor_v1(input_dim1=node_raw_features.shape[1],
                                                  input_dim2=node_raw_features.shape[1],
                                                  hidden_dim=node_raw_features.shape[1], output_dim=1,
                                                  random_projections=None if args.decode_not_rp else random_projections,
                                                  not_encode=args.not_encode)
            model = nn.Sequential(dynamic_backbone, link_predictor)
            logger.info(f'model -> {model}')
            logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                        f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

            # load the saved model
            save_model_name = f"{args.model_name}_seed{run}"
            save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{save_model_name}/"
            early_stopping = EarlyStopping(patience=0, save_model_folder=save_model_folder, logger=logger,
                                           save_model_name=save_model_name,model_name=args.model_name)

            early_stopping.load_checkpoint(model, map_location='cpu')
            model = convert_to_gpu(model, device=args.device)

            # put the node raw messages of memory-based models on device
            if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
                for node_id, node_raw_messages in model[0].memory_bank.node_raw_messages.items():
                    new_node_raw_messages = []
                    for node_raw_message in node_raw_messages:
                        new_node_raw_messages.append((node_raw_message[0].to(args.device), node_raw_message[1]))
                    model[0].memory_bank.node_raw_messages[node_id] = new_node_raw_messages

            if args.model_name in ['DSRD']:
                train_backup = model[0].backup_state()

            # TRANSFER VALIDATION: the checkpoint state is the post-validation state (the training script
            # restores val_backup before saving), so the released flow below scores the validation splits with a state
            # that has already seen them.  For the readout transfer the validation logits must be clean: score
            # validation / new-node validation from the post-train state the patched training script saved, dump
            # them, then restore the checkpoint state and run the released flow unchanged (its own validation calls
            # are not dumped; its test / new-node test calls are).
            if args.model_name in ['DSRD'] and os.environ.get('DYGLIB_DUMP_DIR'):
                import evaluate_models_utils as _emu
                _ts_path = os.path.join(save_model_folder, f'{save_model_name}_train_state.pkl')
                assert os.path.exists(_ts_path), 'no saved post-train state for this checkpoint (%s): retrain with the patched train_link_prediction.py' % _ts_path
                import torch as _torch                  # the script imports torch.nn only
                _ts = _torch.load(_ts_path, map_location=args.device, weights_only=False)
                _train_state = tuple(x.to(args.device) for x in _ts['train_state'])
                _emu.DUMP_ON = True
                model[0].reset_state(_train_state)
                _vl, _vm = evaluate_model_link_prediction(model_name=args.model_name, model=model, neighbor_sampler=full_neighbor_sampler,
                                                          evaluate_idx_data_loader=val_idx_data_loader, evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                          evaluate_data=val_data, loss_func=nn.BCEWithLogitsLoss(), num_neighbors=args.num_neighbors,
                                                          time_gap=args.time_gap, random_projections=None)
                logger.info('transfer validation from the saved post-train state (epoch %d): average_precision %.4f (training-time %.4f)'
                            % (_ts['epoch'], float(np.mean([m['average_precision'] for m in _vm])), _ts['val_metrics']['average_precision']))
                model[0].reset_state(_train_state)
                _nl, _nm = evaluate_model_link_prediction(model_name=args.model_name, model=model, neighbor_sampler=full_neighbor_sampler,
                                                          evaluate_idx_data_loader=new_node_val_idx_data_loader, evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                          evaluate_data=new_node_val_data, loss_func=nn.BCEWithLogitsLoss(), num_neighbors=args.num_neighbors,
                                                          time_gap=args.time_gap, random_projections=None)
                logger.info('transfer new node validation from the saved post-train state: average_precision %.4f (training-time %.4f)'
                            % (float(np.mean([m['average_precision'] for m in _nm])), _ts['new_node_val_metrics']['average_precision']))
                model[0].reset_state(train_backup)      # the checkpoint's own state: the released flow starts from it
                _emu.DUMP_ON = False                    # the released flow's validation calls are not dumped

            loss_func = nn.BCEWithLogitsLoss()




            # evaluate the best model
            logger.info(f'---------get final performance on dataset {args.dataset_name}-------')

            # the saved best model of memory-based models cannot perform validation since the stored memory/ncache/random projections has been updated by validation data
            if args.model_name not in ['JODIE', 'DyRep', 'TGN', 'PINT', 'NAT'] and args.use_random_projection == False:
                
                val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                         model=model,
                                                                         neighbor_sampler=full_neighbor_sampler,
                                                                         evaluate_idx_data_loader=val_idx_data_loader,
                                                                         evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                         evaluate_data=val_data,
                                                                         loss_func=loss_func,
                                                                         num_neighbors=args.num_neighbors,
                                                                         time_gap=args.time_gap,
                                                                         random_projections=random_projections)

                if args.model_name in ['DSRD']:
                    val_backup = model[0].backup_state()
                    model[0].reset_state(train_backup)
                new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                           model=model,
                                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                                           evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                           evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                           evaluate_data=new_node_val_data,
                                                                                           loss_func=loss_func,
                                                                                           num_neighbors=args.num_neighbors,
                                                                                           time_gap=args.time_gap,
                                                                                           random_projections=random_projections)

            # the memory/ncache/random projections in the best model has seen the validation edges, we need to backup the memory for new testing nodes
            if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
                val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()
                if args.model_name == 'PINT':
                    val_backup_matrix_memory = model[0].matrix_memory.backup_memory()
            if args.model_name == 'NAT':
                val_backup_ncache = model[0].backup_ncache()
            if args.use_random_projection:
                val_backup_random_projections = random_projections.backup_random_projections()
            if args.model_name in ['DSRD']:
                model[0].reset_state(val_backup)

            
            if args.model_name in ['DSRD'] and os.environ.get('DYGLIB_DUMP_DIR'):
                import evaluate_models_utils as _emu
                _emu.DUMP_ON = True                     # test / new-node test are dumped
            test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                       model=model,
                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                       evaluate_idx_data_loader=test_idx_data_loader,
                                                                       evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                       evaluate_data=test_data,
                                                                       loss_func=loss_func,
                                                                       num_neighbors=args.num_neighbors,
                                                                       time_gap=args.time_gap,
                                                                       random_projections=random_projections)

            # reload validation memory bank/ncache/random projections for new testing nodes
            if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
                model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)
                if args.model_name == 'PINT':
                    model[0].matrix_memory.reload_memory(val_backup_matrix_memory)
            if args.model_name == 'NAT':
                model[0].reload_ncache(val_backup_ncache)
            if args.use_random_projection:
                random_projections.reload_random_projections(val_backup_random_projections)
            if args.model_name in ['DSRD']:
                model[0].reset_state(val_backup)

            new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                         model=model,
                                                                                         neighbor_sampler=full_neighbor_sampler,
                                                                                         evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                         evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                         evaluate_data=new_node_test_data,
                                                                                         loss_func=loss_func,
                                                                                         num_neighbors=args.num_neighbors,
                                                                                         time_gap=args.time_gap,
                                                                                         random_projections=random_projections)
            # store the evaluation metrics at the current run
            val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

            if args.model_name not in ['JODIE', 'DyRep', 'TGN', 'PINT', 'NAT'] and args.use_random_projection == False:
                logger.info(f'validate loss: {np.mean(val_losses):.4f}')
                for metric_name in val_metrics[0].keys():
                    average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
                    logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
                    val_metric_dict[metric_name] = average_val_metric

                logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
                for metric_name in new_node_val_metrics[0].keys():
                    average_new_node_val_metric = np.mean(
                        [new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
                    logger.info(f'new node validate {metric_name}, {average_new_node_val_metric:.4f}')
                    new_node_val_metric_dict[metric_name] = average_new_node_val_metric

            logger.info(f'test loss: {np.mean(test_losses):.4f}')
            for metric_name in test_metrics[0].keys():
                average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
                logger.info(f'test {metric_name}, {average_test_metric:.4f}')
                test_metric_dict[metric_name] = average_test_metric

            logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
            for metric_name in new_node_test_metrics[0].keys():
                average_new_node_test_metric = np.mean(
                    [new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
                logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
                new_node_test_metric_dict[metric_name] = average_new_node_test_metric

            single_run_time = time.time() - run_start_time
            logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

            if args.model_name not in ['JODIE', 'DyRep', 'TGN', 'PINT', 'NAT'] and args.use_random_projection == False:
                val_metric_all_runs.append(val_metric_dict)
                new_node_val_metric_all_runs.append(new_node_val_metric_dict)
            test_metric_all_runs.append(test_metric_dict)
            new_node_test_metric_all_runs.append(new_node_test_metric_dict)

            # save model result
            if args.model_name not in ['JODIE', 'DyRep', 'TGN', 'PINT', 'NAT'] and args.use_random_projection == False:
                result_json = {
                    "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in
                                         val_metric_dict},
                    "new node validate metrics": {metric_name: f'{new_node_val_metric_dict[metric_name]:.4f}' for
                                                  metric_name in new_node_val_metric_dict},
                    "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in
                                     test_metric_dict},
                    "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for
                                              metric_name in new_node_test_metric_dict}
                }
            else:
                result_json = {
                    "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in
                                     test_metric_dict},
                    "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for
                                              metric_name in new_node_test_metric_dict}
                }
            result_json = json.dumps(result_json, indent=4)

            save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
            os.makedirs(save_result_folder, exist_ok=True)
            save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
            with open(save_result_path, 'w') as file:
                file.write(result_json)
            logger.info(f'save negative sampling results at {save_result_path}')

        # store the average metrics at the end of the log file
        if args.num_runs > 1:
            logger.info(f'-----------metrics over {args.num_runs} runs-----------')
            if args.model_name not in ['JODIE', 'DyRep', 'TGN', 'PINT', 'NAT'] and args.use_random_projection == False:
                for metric_name in val_metric_all_runs[0].keys():
                    logger.info(
                        f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
                    logger.info(
                        f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                        f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

                for metric_name in new_node_val_metric_all_runs[0].keys():
                    logger.info(
                        f'new node validate {metric_name}, {[new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]}')
                    logger.info(
                        f'average new node validate {metric_name}, {np.mean([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]):.4f} '
                        f'± {np.std([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs], ddof=1):.4f}')

            for metric_name in test_metric_all_runs[0].keys():
                logger.info(
                    f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
                logger.info(
                    f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

            for metric_name in new_node_test_metric_all_runs[0].keys():
                logger.info(
                    f'new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
                logger.info(
                    f'average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} '
                    f'± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')
    logger.removeHandler(fh)
    logger.removeHandler(ch)

if __name__ == "__main__":
    evaluate_link_prediction()