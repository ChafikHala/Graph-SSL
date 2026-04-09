from dataclasses import dataclass, replace
from typing import Optional

import torch


@dataclass(frozen=True)
class WLSwAVConfig:
    dataset: str = "cora"
    model: str = "gin"

    hidden_dim: int = 256
    output_dim: int = 128
    dropout: float = 0.0
    encoder_tau: float = 1.0
    heads: int = 4
    n_layers: int = 2

    K: int = 70
    tau: float = 0.2
    eps: float = 0.05
    n_sinkhorn_iters: int = 3
    freeze_prototypes_epochs: int = 1

    W: int = 100
    max_curriculum_level: int = 3

    lambda_uniform: float = 0.05
    lambda_feat: float = 0.0
    hidden_dim_dec: Optional[int] = None

    n_epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0
    batch_size: int = 512
    max_pairs_per_node: int = 5
    log_every: int = 10

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    wl_max_iter: int = 10


def make_wl_swav_cfg(dataset: str) -> WLSwAVConfig:
    return replace(WLSwAVConfig(dataset=dataset))


cfg = make_wl_swav_cfg("cora")
