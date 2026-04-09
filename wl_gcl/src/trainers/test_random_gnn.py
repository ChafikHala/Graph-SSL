from __future__ import annotations

from typing import Dict, List
import random
import numpy as np
from dataclasses import replace

import torch

from wl_gcl.configs.wl_swav import WLSwAVConfig
from wl_gcl.src.data_loader.dataset import load_dataset
from wl_gcl.src.models import get_model
from wl_gcl.src.utils.wl_core import WLHierarchyEngine
from wl_gcl.src.trainers.eval import evaluate_linear_probe


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test_random_gnn(cfg: WLSwAVConfig, seed: int) -> float:
    set_seed(seed)
    device = torch.device(cfg.device)
    dataset = load_dataset(cfg.dataset)
    data = dataset.data.to(device)

    encoder = get_model(
        name=cfg.model,
        input_dim=dataset.num_features,
        hidden_dim=cfg.hidden_dim,
        out_dim=cfg.output_dim,
        dropout=cfg.dropout,
        tau=cfg.encoder_tau,
        num_layers=cfg.n_layers,
        heads=cfg.heads,
    ).to(device)

    encoder.eval()

    acc = evaluate_linear_probe(encoder, data, dataset.num_classes, device)
    print(f"[Random GNN | seed={seed}] acc = {acc:.4f}")
    return acc


def run_multi_seed(
    cfg: WLSwAVConfig,
    seeds: List[int] = [0, 1, 2, 3, 4],
) -> Dict[str, float]:
    accs = [test_random_gnn(cfg, seed) for seed in seeds]
    mean = float(np.mean(accs))
    std  = float(np.std(accs))
    print(f"\n[Random GNN | {cfg.model.upper()} | {cfg.dataset}] "
          f"mean={mean:.4f}  std={std:.4f}  "
          f"over {len(seeds)} seeds: {[f'{a:.4f}' for a in accs]}")
    return {"mean": mean, "std": std, "accs": accs}


if __name__ == "__main__":
    cfg = WLSwAVConfig()
    cfg = replace(cfg, dataset="cora", model="gcn")  
    print(f"Dataset: {cfg.dataset} | Model: {cfg.model}")
    run_multi_seed(cfg, seeds=[0, 1, 2, 3, 4])