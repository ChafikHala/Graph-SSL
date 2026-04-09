from __future__ import annotations

from pathlib import Path
from typing import Dict
import random
import numpy as np

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from wl_gcl.configs.wl_swav import WLSwAVConfig
from wl_gcl.src.data_loader.dataset import load_dataset
from wl_gcl.src.losses.wl_swav_loss import WLSwAVLoss
from wl_gcl.src.models import get_model
from wl_gcl.src.trainers.wl_swav_trainer import WLSwAVTrainer
from wl_gcl.src.utils.wl_core import WLHierarchyEngine
from wl_gcl.src.trainers.eval import evaluate_linear_probe


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def train_wl_swav(cfg: WLSwAVConfig) -> Dict[str, float]:
    device = torch.device(cfg.device)
    dataset = load_dataset(cfg.dataset)
    data = dataset.data.to(device)

    nodes = list(range(data.num_nodes))
    edges = data.edge_index.t().tolist()

    wl_engine = WLHierarchyEngine(nodes, edges)
    wl_engine.build_wl_tree(
        max_iterations=cfg.wl_max_iter,
        force_convergence=(data.num_nodes < 1000),
    )

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

    loss_fn = WLSwAVLoss(
        K=cfg.K,
        tau=cfg.tau,
        eps=cfg.eps,
        n_sinkhorn_iters=cfg.n_sinkhorn_iters,
        freeze_prototypes_epochs=cfg.freeze_prototypes_epochs,
        embedding_dim=cfg.output_dim,
        feature_dim=dataset.num_features,
        lambda_uniform=cfg.lambda_uniform,
        lambda_feat=cfg.lambda_feat,
        hidden_dim_dec=cfg.hidden_dim_dec,
    ).to(device)
    loss_fn.initialize_prototypes(device)

    optimizer = Adam(
        list(encoder.parameters()) + list(loss_fn.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)

    trainer = WLSwAVTrainer(
        model=encoder,
        wl_engine=wl_engine,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=cfg,
    )

    result = trainer.train(data, dataset, cfg.n_epochs, cfg.log_every)
    final_stats = result.pop("final_stats", {})
    best_state = result.pop("best_state", None)

    if best_state is not None:
        out_dir = Path("runs/wl_swav") / cfg.dataset / cfg.model
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / "best_encoder.pt"
        torch.save(
            {
                "encoder_state_dict": best_state,
                "best_accuracy": result["best_accuracy"],
                "cfg": cfg.__dict__,
            },
            ckpt_path,
        )
    else:
        ckpt_path = None

    return {
        "dataset": result.get("dataset", cfg.dataset),
        "best_accuracy": result["best_accuracy"],
        "epochs": cfg.n_epochs,
        "best_ckpt_path": str(ckpt_path) if ckpt_path is not None else None,
        "ablation_stats": final_stats,
    }

