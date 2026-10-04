import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Literal, Tuple

from utils.utils import NeighborSampler
from models.modules import TimeEncoder


class DSRDBlock(nn.Module):
    """
    DSRDBlock (DSRD core): computes edge-level pulses and aggregates to node states.
    Simplified to parallel mode; slots and chunkwise paths are removed for clarity.
    """

    def __init__(
        self,
        in_dim: int,
        head_dim: int,
        edge_feat_dim: int = 0,
        heads: int = 2,
        dropout: float = 0.1,
        concat: bool = False,
    ):
        super().__init__()
        self.Fin = in_dim
        self.C = head_dim
        self.H = heads
        self.edge_feat_dim = edge_feat_dim if edge_feat_dim > 0 else in_dim
        self.concat = concat

        self.lin_q = nn.Linear(in_dim, heads * head_dim, bias=False)
        self.lin_k = nn.Linear(in_dim, heads * head_dim, bias=False)
        self.lin_v = nn.Linear(in_dim, heads * head_dim, bias=False)
        self.lin_edge = nn.Linear(self.edge_feat_dim, heads * head_dim, bias=True)

        self.state_decay_logits = nn.Parameter(torch.randn(heads), requires_grad=True)
        self.time_decay_rate = nn.Parameter(torch.randn(heads), requires_grad=True)
        self.time_sensitivity = nn.Parameter(torch.zeros(heads), requires_grad=True)

        # Pre-normalization to stabilize Q/K/V scales
        self.pre_norm_node = nn.LayerNorm(in_dim)
        self.pre_norm_edge = nn.LayerNorm(self.edge_feat_dim)

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_q.weight)
        nn.init.xavier_uniform_(self.lin_k.weight)
        nn.init.xavier_uniform_(self.lin_v.weight)
        if self.lin_q.bias is not None:
            nn.init.zeros_(self.lin_q.bias)
        if self.lin_k.bias is not None:
            nn.init.zeros_(self.lin_k.bias)
        if self.lin_v.bias is not None:
            nn.init.zeros_(self.lin_v.bias)

    def compute_edge_decay(self, delta_t_e):
        """
        Compute time-based decay for edges using learnable parameters.

        Args:
            delta_t_e: Time differences for edges [E]

        Returns:
            time_decay: Decay weights per head [E, H]
        """
        lambda_h = F.softplus(self.time_decay_rate)  # Positive decay rate per head
        alpha_h = torch.sigmoid(self.time_sensitivity)  # Power exponent in [0,1] per head
        exp_arg = -(torch.pow(delta_t_e.unsqueeze(-1), alpha_h.unsqueeze(0)) * lambda_h.unsqueeze(0))
        time_decay = torch.exp(exp_arg)
        return time_decay

    def _prepare_strengths(self, key_e, query_n, dst):
        """
        Compute attention strengths between query nodes and keys.

        Args:
            key_e: Edge keys [E, H, C]
            query_n: Node queries [N, H, C]
            dst: Destination node indices [E]

        Returns:
            fused_strength_eh: Attention weights in [0,1] per edge and head [E, H]
        """
        C = key_e.size(-1)
        qk = (query_n[dst] * key_e).sum(dim=-1) / math.sqrt(C)  # Scaled dot-product
        fused_strength_eh = torch.sigmoid(qk)  # Normalize to [0,1]
        return fused_strength_eh

    def message(self, key_e, val_e, fused_strength_eh, edge_feats):
        """
        Compute edge messages with optional edge features and temporal decay.

        Args:
            key_e: Edge keys [E, H, C]
            val_e: Edge values [E, H, C]
            fused_strength_eh: Attention strengths [E, H]
            edge_feats: Dictionary with optional edge_attr, edge_center_dt, edge_batch_index

        Returns:
            phi_e: Weighted key-value products [E, H, C, C]
        """
        E, H, C = key_e.shape
        edge_attr_e = edge_feats.get("edge_attr", None)
        edge_center_dt = edge_feats.get("edge_center_dt", None)
        edge_batch_index = edge_feats.get("edge_batch_index", None)

        # Incorporate edge features if available
        if edge_attr_e is not None:
            edge_attr_e = self.pre_norm_edge(edge_attr_e)
            edge_attr_e = self.lin_edge(edge_attr_e)
            edge_k = self.lin_k(edge_attr_e).view(E, H, C)
            edge_v = self.lin_v(edge_attr_e).view(E, H, C)
            key_e = key_e + edge_k
            val_e = val_e + edge_v

        # Compute outer product of keys and values
        kv = key_e.unsqueeze(-1) @ val_e.unsqueeze(-2)  # [E, H, C, C]

        # Apply temporal decay and attention weights
        edge_decay_eh = self.compute_edge_decay(delta_t_e=edge_center_dt)
        weight_eh = fused_strength_eh * edge_decay_eh
        weight_eh = self.dropout(weight_eh)

        # Normalize weights per batch sample if batch indices provided
        if edge_batch_index is not None and edge_batch_index.numel() > 0:
            B = int(edge_batch_index.max().item()) + 1
            norm = torch.zeros(B, H, device=key_e.device, dtype=weight_eh.dtype)
            norm.index_add_(0, edge_batch_index, weight_eh)
            norm = norm[edge_batch_index] + 1e-8
            weight_eh = weight_eh / norm.clamp_min(1.0)

        # Weight the KV products
        phi_e = weight_eh.unsqueeze(-1).unsqueeze(-1) * kv
        return phi_e

    def aggregate(self, phi_e, dst, num_nodes):
        """
        Aggregate edge messages to destination nodes.

        Args:
            phi_e: Edge messages [E, H, C, C]
            dst: Destination node indices [E]
            num_nodes: Total number of nodes in the batch

        Returns:
            Aggregated state updates [num_nodes, H, C, C]
        """
        E, H, C, _ = phi_e.shape
        flat = phi_e.view(E, H, C * C)
        out = torch.zeros(num_nodes, H, C * C, device=phi_e.device, dtype=phi_e.dtype)
        out.index_add_(0, dst, flat)  # Sum messages to the same destination
        return out.view(num_nodes, H, C, C)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None,
        edge_feats: Optional[Dict[str, torch.Tensor]] = None,
        batch_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of DSRDBlock: compute state updates and node outputs.

        Args:
            x: Node features [N, in_dim]
            edge_index: Edge connectivity [2, E], where edge_index[0] = src, edge_index[1] = dst
            prev_state: Previous node states [N, H, C, C] (optional)
            edge_feats: Dictionary of edge features (optional)
            batch_index: Batch assignment for each node [N] (optional)

        Returns:
            Dictionary with "out" (node outputs) and "state" (updated states)
        """
        N = x.size(0)
        H, C = self.H, self.C
        src, dst = edge_index

        # Project node features to Q, K, V
        h = self.pre_norm_node(x)
        Q = self.lin_q(h).view(N, H, C)
        Kx = self.lin_k(h).view(N, H, C)
        Vx = self.lin_v(h).view(N, H, C)

        # Extract edge keys and values from source nodes
        key_e = Kx[src]
        val_e = Vx[src]

        # Compute attention strengths and messages
        fused_strength_eh = self._prepare_strengths(key_e, Q, dst)  # [E, H]
        phi_e = self.message(key_e, val_e, fused_strength_eh, edge_feats)
        S_delta = self.aggregate(phi_e, dst=dst, num_nodes=N)  # [N, H, C, C]

        # Initialize state if not provided
        if prev_state is None:
            prev_state = torch.zeros_like(S_delta)

        # Update state with learnable decay parameter
        gamma_h = torch.sigmoid(self.state_decay_logits).view(1, H, 1, 1)
        state_new = gamma_h * prev_state + (1 - gamma_h) * S_delta

        # Compute output by multiplying state with queries
        out_vec = torch.einsum("nhij,nhj->nhi", state_new, Q)
        if self.concat:
            out = out_vec.view(N, -1)  # Concatenate all heads
        else:
            out = out_vec.mean(dim=1)  # Average across heads

        return {"out": out, "state": state_new}


class DSRD(nn.Module):
    """
    Unified memory encoder: no multi-hop sampling, combines KV propagation
    over current batch edges and temporal walk updates.
    """

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 channel_embedding_dim: int, time_feat_dim: int = 0, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1,
                 num_neighbors: int = 20, device: str = 'cpu'):
        
        super().__init__()
        self.device = torch.device(device)
        self.channel_embedding_dim = channel_embedding_dim
        assert channel_embedding_dim % num_heads == 0, 'channel_embedding_dim must be divisible by num_heads.'
        self.head_dim = channel_embedding_dim // num_heads
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.neighbor_sampler = neighbor_sampler
        self.num_neighbors = num_neighbors

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(self.device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(self.device)
        
        self.num_nodes = self.node_raw_features.shape[0]
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
   
        self.node_proj = nn.Linear(self.node_feat_dim, channel_embedding_dim, bias=True)
        self.time_encoder = TimeEncoder(time_dim=self.edge_feat_dim)
        self.out_proj = nn.Linear(channel_embedding_dim, self.node_feat_dim)

        self.blocks = nn.ModuleList([
            DSRDBlock(in_dim=channel_embedding_dim,head_dim=self.head_dim,edge_feat_dim=self.edge_feat_dim, heads=self.num_heads, dropout=dropout, concat=True)
            for _ in range(num_layers)
        ])
        self.ffn=nn.ModuleList([
            ResidualFFN(model_dim=channel_embedding_dim)
            for _ in range(num_layers)
        ])
       
        self.time_decay_weight = nn.Parameter(
            torch.full((self.num_layers,), 10.0, device=self.device, dtype=torch.float),
            requires_grad=True,
            )
        
        self.reset_state()

    @torch.no_grad()
    def backup_state(self):
        """
        Create a backup of the current model state for later restoration.
        Used for proper evaluation on temporal data where state needs to be reset.

        Returns:
            Tuple of (state, now_time) tensors
        """
        return (
            self.state.detach().clone(),
            self.now_time.detach().clone(),
        )

    @torch.no_grad()
    def reset_state(self, backup=None):
        """
        Reset or restore the model state.

        Args:
            backup: Optional tuple of (state, now_time) to restore from.
                   If None, initializes to zero state.
        """
        if backup is None:
            if not hasattr(self, "state"):
                # Initialize state on first call
                self.state = nn.Parameter(
                    torch.zeros(self.num_nodes, self.num_layers, self.num_heads, self.head_dim, self.head_dim,
                        device=self.device),requires_grad=False,
                )
                self.now_time = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=False)
            else:
                # Reset existing state to zero
                self.state.data.zero_()
                self.now_time.data.zero_()
        else:
            # Restore from backup
            state_backup, now_time_backup = backup
            self.state.data.copy_(state_backup)
            self.now_time.data.copy_(now_time_backup)


    def _build_batch_graph(
                        self,
                        node_ids: np.ndarray,
                        node_interact_times: np.ndarray
                        ):
        """
        Construct the batch subgraph by sampling historical neighbors for each center node.

        This method builds a temporal graph structure for a batch of nodes by:
        1. Sampling historical neighbors for each center node
        2. Creating edge connectivity between neighbors and centers
        3. Computing temporal edge features and time deltas
        4. Mapping global node IDs to batch-local indices

        Args:
            node_ids: Array of center node IDs [B]
            node_interact_times: Interaction timestamps for center nodes [B]

        Returns:
            unique_nodes_all: All unique nodes in the batch (centers + neighbors) [N_sub]
            edge_index: Edge connectivity [2, E] (src -> dst)
            edge_feats: Dictionary with edge_attr, edge_center_dt, edge_batch_index
            center_idx: Batch-local indices of center nodes [B]
            batch_index: Batch assignment for each node [N_sub]
        """
        assert self.neighbor_sampler is not None, "NeighborSampler is not set. Please call set_neighbor_sampler()."
        device, dtype_f, ns = self.device, torch.float32, self.neighbor_sampler
        node_ids = np.asarray(node_ids, dtype=np.int64)
        node_interact_times = np.asarray(node_interact_times, dtype=np.float32)
        B = node_ids.shape[0]
        assert B == node_interact_times.shape[0]
        
        # Convert center nodes to tensors
        centers = torch.as_tensor(node_ids, device=device, dtype=torch.long)          # [B]
        center_times = torch.as_tensor(node_interact_times, device=device, dtype=dtype_f)

        # Sample historical neighbors for each center node (returns [B, K] arrays)
        nbr_nodes_np, nbr_eids_np, nbr_times_np = ns.get_historical_neighbors(
            node_ids=node_ids,
            node_interact_times=node_interact_times,
            num_neighbors=self.num_neighbors,
        )
        nbr_nodes_np = np.asarray(nbr_nodes_np, dtype=np.int64)
        nbr_eids_np = np.asarray(nbr_eids_np, dtype=np.int64)
        nbr_times_np = np.asarray(nbr_times_np, dtype=np.float32)
        assert nbr_nodes_np.shape == nbr_eids_np.shape == nbr_times_np.shape
        assert nbr_nodes_np.shape[0] == B
        B2, K = nbr_nodes_np.shape
        assert B2 == B
        
        # Convert neighbor data to tensors
        nbr_nodes = torch.as_tensor(nbr_nodes_np, device=device, dtype=torch.long)    # [B, K]
        nbr_eids = torch.as_tensor(nbr_eids_np, device=device, dtype=torch.long)      # [B, K]
        nbr_times = torch.as_tensor(nbr_times_np, device=device, dtype=dtype_f)       # [B, K]

        # Identify valid (non-padded) neighbors
        pad_mask = (nbr_nodes == 0) & (nbr_eids == 0)
        valid_mask = ~pad_mask                                                        # [B, K]
        
        # Critical time assertion: for valid neighbors, must satisfy t_neighbor <= t_center
        if valid_mask.any():
            center_times_expanded = center_times.view(B, 1).expand(B, K)
            assert torch.all(
                nbr_times[valid_mask] <= center_times_expanded[valid_mask] + 1e-6
            ), "get_historical_neighbors returned neighbors with timestamps later than center nodes - potential time leakage"
            
        # Flatten valid neighbors and track which center they belong to
        if valid_mask.any():
            nbr_nodes_flat = nbr_nodes[valid_mask]                                    # [E]
            nbr_eids_flat = nbr_eids[valid_mask]
            nbr_times_flat = nbr_times[valid_mask]
            batch_ids = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(B, K)
            center_for_edge = batch_ids[valid_mask]                                   # [E] - which center each edge belongs to
        else:
            # Handle case with no valid neighbors
            nbr_nodes_flat = torch.empty(0, device=device, dtype=torch.long)
            nbr_eids_flat = torch.empty(0, device=device, dtype=torch.long)
            nbr_times_flat = torch.empty(0, device=device, dtype=dtype_f)
            center_for_edge = torch.empty(0, device=device, dtype=torch.long)
        
        # Combine center nodes and their neighbors into a single node list
        unique_nodes_all = torch.cat([centers, nbr_nodes_flat], dim=0)                # [N_sub]
        N_sub = unique_nodes_all.numel()

        # Create batch index: which batch sample each node belongs to
        batch_index_centers = torch.arange(B, device=device, dtype=torch.long)
        batch_index_neighbors = center_for_edge                                      # [E]
        batch_index = torch.cat([batch_index_centers, batch_index_neighbors], dim=0)
        assert batch_index.shape[0] == N_sub

        # ROW-FIX: every center keeps its OWN row and every neighbour occurrence
        # its own row (no last-occurrence collision across centers of the batch)
        center_idx = torch.arange(B, device=device, dtype=torch.long)               # [B]
        assert torch.equal(unique_nodes_all[center_idx], centers)
        
        # Handle case with no edges
        if nbr_nodes_flat.numel() == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_feats = {
                "edge_attr": torch.zeros((0, self.edge_feat_dim), dtype=dtype_f, device=device),
                "edge_center_dt": torch.zeros(0, dtype=dtype_f, device=device),
                "edge_batch_index": torch.zeros(0, dtype=torch.long, device=device),
            }
            return unique_nodes_all, edge_index, edge_feats, center_idx, batch_index
        
        # Build edges: neighbor -> center (rows are per-occurrence)
        E_n = nbr_nodes_flat.numel()
        src_batch = B + torch.arange(E_n, device=device, dtype=torch.long)
        dst_batch = center_for_edge

        edge_index = torch.stack([src_batch, dst_batch], dim=0)                      # [2, E]

        # Compute time deltas and encode them
        dt_raw = center_times[center_for_edge] - nbr_times_flat                      # [E]
        dt_all = torch.log1p(dt_raw)  # Log transform for stability

        # Combine raw edge features with time encoding
        base_edge_attr = self.edge_raw_features[nbr_eids_flat]
        dt_emb = self.time_encoder(dt_all.unsqueeze(1)).squeeze()
        edge_attr = base_edge_attr + dt_emb
        
        edge_feats = {
            "edge_attr": edge_attr.to(dtype_f),
            "edge_center_dt": dt_all.to(dtype_f),
            "edge_batch_index": center_for_edge.to(torch.long),
        }

        return unique_nodes_all, edge_index, edge_feats, center_idx, batch_index

    def compute_src_dst_node_temporal_embeddings(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        batch_edge_ids: np.ndarray,
        node_interact_times: np.ndarray,
        edges_are_positive: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute temporal embeddings for source and destination nodes.

        This is the main forward pass for link prediction. It:
        1. Builds a batch subgraph with historical neighbors
        2. Applies temporal decay to node states
        3. Propagates information through DSRD blocks
        4. Updates node states if edges are positive (training mode)
        5. Returns embeddings for source and destination nodes

        Args:
            src_node_ids: Source node IDs [B]
            dst_node_ids: Destination node IDs [B]
            batch_edge_ids: Edge IDs for the batch [B]
            node_interact_times: Interaction timestamps [B]
            edges_are_positive: If True, update node states after forward pass

        Returns:
            Tuple of (src_embeddings, dst_embeddings) each of shape [B, node_feat_dim]
        """

        global_node_idx, edge_index, edge_feats, center_idx, batch_index = self._build_batch_graph(
            node_ids=np.concatenate([src_node_ids, dst_node_ids]),
            node_interact_times=np.concatenate([node_interact_times,node_interact_times])
        )

        # If there are no nodes or edges, directly return original features
        if global_node_idx.numel() == 0 or edge_index.numel() == 0:
            src_tensor = torch.as_tensor(src_node_ids, device=self.device, dtype=torch.long)
            dst_tensor = torch.as_tensor(dst_node_ids, device=self.device, dtype=torch.long)
            return self.node_raw_features[src_tensor], self.node_raw_features[dst_tensor]

        # Project node features to embedding space
        # 1-hop expansion of all nodes (including duplicates)
        x = self.node_proj(self.node_raw_features[global_node_idx])

        # Compute temporal decay for state evolution
        t = torch.as_tensor(node_interact_times, device=self.device, dtype=torch.float32)
        next_time = t[-1]
        dt_move = torch.log1p((next_time - self.now_time).clamp_min(0.0))
        assert torch.isfinite(dt_move).all()

        # Propagate through DSRD blocks
        # batch_index has the same length as nodes_all, indicating which center sample each node belongs to
        for l, block in enumerate(self.blocks):
            # Apply layer-specific temporal decay to states
            tau = F.softplus(self.time_decay_weight[l])
            decay = torch.exp(-tau * dt_move * float(l + 1)).view(1, 1, 1, 1)
            prev_state = self.state[global_node_idx, l] * decay

            # Forward through retention block
            out = block(x, edge_index, prev_state=prev_state, edge_feats=edge_feats, batch_index=batch_index)
            x = out["out"] + x  # Residual connection
            x = self.ffn[l](x)  # Feed-forward network

            # Update states for positive edges (training mode)
            if edges_are_positive:
                self.state[global_node_idx, l].copy_(out["state"].detach())

        # Project back to original feature dimension
        x = self.out_proj(x)
      
        # Extract embeddings for source and destination nodes
        src_idx = center_idx[:len(src_node_ids)]
        dst_idx = center_idx[len(src_node_ids):]
        src_emb = x[src_idx]
        dst_emb = x[dst_idx]

        # Perform temporal walk update if edges are positive
        if edges_are_positive:
            self.update(src_node_ids, dst_node_ids, node_interact_times)

        return src_emb, dst_emb
       
    def update(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        Perform temporal walk update to propagate states across layers.

        This implements a diffusion-like process where states from lower layers
        are propagated to higher layers through temporal edges. For each edge (src, dst),
        we exchange state information between connected nodes with temporal decay.

        Args:
            src_node_ids: Source node IDs [B]
            dst_node_ids: Destination node IDs [B]
            node_interact_times: Interaction timestamps [B]
        """
        if len(src_node_ids) == 0 or self.num_layers <= 1:
            return

        src = torch.as_tensor(src_node_ids, device=self.device, dtype=torch.long)
        dst = torch.as_tensor(dst_node_ids, device=self.device, dtype=torch.long)
        times = torch.as_tensor(node_interact_times, device=self.device, dtype=torch.float)
        next_time_t = torch.tensor(float(node_interact_times[-1]), device=self.device, dtype=torch.float)

        delta_event = torch.log1p(next_time_t - times)  # [B] - time since each event

        # Propagate states from lower to higher layers (backward pass)
        for l in range(self.num_layers - 1, 1, -1):
            # Compute temporal decay weight
            tau = F.softplus(self.time_decay_weight[l])
            w = torch.exp(-tau * delta_event * float(l + 1)).view(-1, 1, 1, 1)

            # Get states from previous layer
            prev = self.state[:, l - 1]

            # Aggregate messages from neighbors
            msg_sum = torch.zeros_like(self.state[:, l])
            weight_sum = torch.zeros(self.num_nodes, 1, 1, 1, device=self.device)

            # Exchange states: src receives from dst's prev layer, and vice versa
            msg_sum.index_add_(0, src, w * prev[dst])
            msg_sum.index_add_(0, dst, w * prev[src])
            weight_sum.index_add_(0, src, w)
            weight_sum.index_add_(0, dst, w)

            # Compute weighted average
            mean_msg = msg_sum / weight_sum.clamp_min(1.0)

            # Update current layer state with learnable mixing weight
            beta = torch.sigmoid(tau).view(1, 1, 1, 1)
            new_state = beta * self.state[:, l] + (1 - beta) * mean_msg

            assert torch.isfinite(new_state).all(), f"NaN in walk update at layer {l}"
            self.state[:, l].copy_(new_state)
            with torch.no_grad():
                self.state[:, l].copy_(new_state)

        # Update global time tracker
        with torch.no_grad():
            self.now_time.data = next_time_t
        
    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()
        
        
class ResidualFFN(nn.Module):
    """
    Simple residual feed-forward network: LayerNorm -> FFN -> Dropout + Residual.
    """

    def __init__(self, model_dim: int, hidden_mult: float = 2.0, dropout: float = 0.1, use_layernorm: bool = True):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.norm = nn.LayerNorm(model_dim) if use_layernorm else nn.Identity()
        hidden_dim = int(model_dim * hidden_mult)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.ffn(h)
        return x + h
