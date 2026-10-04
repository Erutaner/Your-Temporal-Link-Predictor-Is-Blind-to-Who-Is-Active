import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils.metrics import get_link_prediction_metrics
from utils.utils import NegativeEdgeSampler, NeighborSampler
from utils.DataLoader import Data


def evaluate_model_link_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module,
                                   num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate GRN on the link prediction task (random / historical / inductive negatives); the positive call of a
    batch precedes its negative call (the negative call scores against the same committed state and chunk context)
    """
    assert model_name == 'GRN'
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    model[0].set_neighbor_sampler(neighbor_sampler)
    model.eval()

    with torch.no_grad():
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices], evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                                  batch_src_node_ids=batch_src_node_ids,
                                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                                  current_batch_start_time=batch_node_interact_times[0],
                                                                                                  current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))
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

            evaluate_losses.append(loss.item())
            evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics


def replay_positive_events(model: nn.Module, idx_data_loader: DataLoader, data: Data):
    """
    stream the positive events of `data` through the model state without scoring (used to rebuild the state of a
    loaded checkpoint before evaluation)
    """
    model.eval()
    with torch.no_grad():
        for data_indices in tqdm(idx_data_loader, ncols=120, desc='replay'):
            data_indices = data_indices.numpy()
            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=data.src_node_ids[data_indices],
                                                              dst_node_ids=data.dst_node_ids[data_indices],
                                                              node_interact_times=data.node_interact_times[data_indices],
                                                              batch_edge_ids=data.edge_ids[data_indices],
                                                              edges_are_positive=True, need_scores=False)
