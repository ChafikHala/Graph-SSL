# src/contrastive/losses.py
from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_sim_matrix(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between two embedding batches.

    Args:
        z1, z2: (N, d)

    Returns:
        sim: (N, N)
    """
    return torch.matmul(z1, z2.T)


def info_nce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    Standard SimCLR/InfoNCE contrastive loss.

    Positive for each i is (z1[i], z2[i]).
    Negatives: all other elements of the batch.

    Args:
        z1, z2: shape (N, d), already L2-normalized.
        temperature: temperature scaling
        High temperature : Softmax becomes smoother, Treats similarities more uniformly
        Low temperature (τ ↓) : Softmax becomes very sharp, Emphasizes the highest similarity scores

    Returns:
        loss: scalar tensor
    """
    assert z1.shape == z2.shape

    N = z1.size(0)

    # Pairwise similarity
    sim = cosine_sim_matrix(z1, z2)                      # (N, N)
    sim = sim / temperature

    # Positive similarities are on diagonal
    pos_sim = torch.diag(sim)

    # Denominator: sum over all columns
    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1)

    loss = -torch.log(torch.exp(pos_sim) / denom)
    return loss.mean()


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    NT-Xent contrastive loss (SimCLR / GraphCL standard).

    Positive for each i is (z1[i], z2[i]).
    Negatives: all other elements of the batch.

    Args:
        z1, z2: shape (N, d), already L2-normalized.
        temperature: temperature scaling
        High temperature : Softmax becomes smoother, Treats similarities more uniformly
        Low temperature (τ ↓) : Softmax becomes very sharp, Emphasizes the highest similarity scores

    Returns:
        loss: scalar tensor
    """

    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)

    sim = cosine_sim_matrix(z, z)
    sim = sim / temperature

    mask = torch.eye(2*N, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)

    pos_idx = torch.arange(N, device=z.device)
    pos_idx = torch.cat([pos_idx + N, pos_idx])

    log_prob = torch.log_softmax(sim, dim=1)

    loss = -log_prob[torch.arange(2*N), pos_idx]
    return loss.mean()


