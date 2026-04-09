from __future__ import annotations

from __future__ import annotations

import random
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from wl_gcl.src.utils.wl_core import WLHierarchyEngine


class WLSwAVLoss(nn.Module):
    """Swapped prediction loss plus auxiliary regularizers."""

    def __init__(
        self,
        K: int,
        tau: float,
        eps: float,
        n_sinkhorn_iters: int,
        freeze_prototypes_epochs: int,
        embedding_dim: int,
        feature_dim: int,
        lambda_uniform: float = 0.5,
        lambda_feat: float = 0.1,
        hidden_dim_dec: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.K = K
        self.tau = tau
        self.eps = eps
        self.n_sinkhorn_iters = n_sinkhorn_iters
        self.freeze_prototypes_epochs = freeze_prototypes_epochs
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.lambda_uniform = lambda_uniform
        self.lambda_feat = lambda_feat
        self.prototypes: Optional[nn.Parameter] = None

        hidden_dec = hidden_dim_dec or max(1, embedding_dim // 2)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dec),
            nn.ReLU(),
            nn.Linear(hidden_dec, feature_dim),
        )

    def _initialize_prototypes(self, device: torch.device) -> None:
        prototypes = torch.randn(self.K, self.embedding_dim, device=device)
        prototypes = F.normalize(prototypes, dim=1)
        self.prototypes = nn.Parameter(prototypes)

    def initialize_prototypes(self, device: torch.device) -> None:
        if self.prototypes is None:
            self._initialize_prototypes(device)

    def sinkhorn(self, M: torch.Tensor) -> torch.Tensor:
        """Sinkhorn-Knopp normalization to obtain soft codes."""
        Q = torch.exp(M)
        num_rows = Q.size(0)
        eps = 1e-12

        for _ in range(self.n_sinkhorn_iters):
            col_sum = Q.sum(dim=0, keepdim=True)
            Q = Q / ((col_sum + eps) * self.K)
            row_sum = Q.sum(dim=1, keepdim=True)
            Q = Q / ((row_sum + eps) * num_rows)

        return Q

    def forward(
        self,
        z_v: torch.Tensor,
        z_u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.prototypes is None:
            self._initialize_prototypes(z_v.device)

        if z_v.size(0) != z_u.size(0):
            raise ValueError("z_v and z_u must contain the same number of samples")

        batch_size = z_v.size(0)
        prototypes = self.prototypes

        z_concat = torch.cat([z_v, z_u], dim=0)
        scores = torch.matmul(z_concat, prototypes.t()) / self.eps

        q = self.sinkhorn(scores).detach()
        q_v = q[:batch_size]
        q_u = q[batch_size:]

        logits_v = torch.matmul(z_v, prototypes.t()) / self.tau
        logits_u = torch.matmul(z_u, prototypes.t()) / self.tau

        p_v = F.softmax(logits_v, dim=-1)
        p_u = F.softmax(logits_u, dim=-1)

        eps_log = 1e-8
        loss_v = - (q_u * torch.log(p_v + eps_log)).sum(dim=-1).mean()
        loss_u = - (q_v * torch.log(p_u + eps_log)).sum(dim=-1).mean()

        return (loss_v + loss_u) * 0.5, q_v

    def uniformity_loss(self, Z: torch.Tensor, sample_size: int = 512) -> torch.Tensor:
        N = Z.size(0)
        if N == 0:
            return torch.tensor(0.0, device=Z.device)

        M = min(N, sample_size)
        idx = torch.randperm(N, device=Z.device)[:M]
        Zs = Z[idx]

        delta = Zs.unsqueeze(1) - Zs.unsqueeze(0)
        sqdist = delta.pow(2).sum(dim=-1)
        exp = torch.exp(-2 * sqdist)
        mask = torch.eye(M, device=Z.device, dtype=torch.bool)
        exp = exp.masked_fill(mask, 0.0)

        value = exp.sum() / (M * M)
        value = torch.clamp(value, min=1e-12)
        return torch.log(value)

    def repulsion_loss(
        self,
        Z: torch.Tensor,
        level: int,
        wl_engine: "WLHierarchyEngine",
        sample_pairs: int = 512,
    ) -> torch.Tensor:
        N = Z.size(0)
        if N == 0:
            return torch.tensor(0.0, device=Z.device)

        nodes = wl_engine.nodes
        if not nodes:
            return torch.tensor(0.0, device=Z.device)

        max_pairs = min(sample_pairs, N)
        pairs = []
        attempts = 0
        while len(pairs) < max_pairs and attempts < max_pairs * 10:
            i = random.randrange(N)
            j = random.randrange(N)
            if i == j:
                attempts += 1
                continue
            cid_i = wl_engine.get_cluster_id(nodes[i], level)
            cid_j = wl_engine.get_cluster_id(nodes[j], level)
            if cid_i is None or cid_j is None or cid_i == cid_j:
                attempts += 1
                continue
            pairs.append((i, j))
            attempts += 1

        if not pairs:
            return torch.tensor(0.0, device=Z.device)

        idx_v = torch.tensor([p[0] for p in pairs], device=Z.device, dtype=torch.long)
        idx_u = torch.tensor([p[1] for p in pairs], device=Z.device, dtype=torch.long)
        dots = (Z[idx_v] * Z[idx_u]).sum(dim=-1)
        return dots.mean()

    def feature_loss(self, Z: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        recon = self.decoder(Z)
        cos = F.cosine_similarity(X, recon, dim=1)
        return (1.0 - cos).mean()

    def normalize_prototypes(self) -> None:
        if self.prototypes is None:
            return
        with torch.no_grad():
            self.prototypes.data.copy_(F.normalize(self.prototypes.data, dim=1))
