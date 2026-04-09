from __future__ import annotations

import copy
import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.optim
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Data

from wl_gcl.src.losses.wl_swav_loss import WLSwAVLoss
from wl_gcl.src.trainers.eval import evaluate_linear_probe
from wl_gcl.src.utils.wl_core import WLHierarchyEngine

logger = logging.getLogger(__name__)


class WLSwAVTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        wl_engine: WLHierarchyEngine,
        loss_fn: WLSwAVLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: Any,
    ) -> None:
        self.model = model
        self.wl_engine = wl_engine
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = torch.device(config.device)
        self._pair_cache: Dict[int, List[Tuple[int, int]]] = {}
        self._last_q_v: Optional[torch.Tensor] = None

    def build_level_pairs(self, level: int) -> List[Tuple[int, int]]:
        if level in self._pair_cache:
            return self._pair_cache[level]

        max_pairs = getattr(self.config, "max_pairs_per_node", 5)
        buckets: Dict[str, List[int]] = {}
        for node in self.wl_engine.nodes:
            cid = self.wl_engine.get_cluster_id(node, level)
            if cid is None:
                continue
            buckets.setdefault(cid, []).append(self.wl_engine.node2idx[node])

        pairs: List[Tuple[int, int]] = []
        for members in buckets.values():
            if len(members) < 2:
                continue
            for anchor in members:
                others = [node for node in members if node != anchor]
                sample_size = min(max_pairs, len(others))
                sampled = random.sample(others, sample_size) if len(others) > sample_size else list(others)
                for partner in sampled:
                    pairs.append((anchor, partner))

        self._pair_cache[level] = pairs
        return pairs

    def get_current_level(self, epoch: int) -> int:
        max_level = max(1, getattr(self.config, "max_curriculum_level", 3))
        warmup = max(1, getattr(self.config, "W", 100))
        step = max(1, warmup // max_level)
        return min(max_level, 1 + epoch // step)

    def _select_level_with_pairs(self, requested_level: int) -> Tuple[int, List[Tuple[int, int]]]:
        level = requested_level
        while level > 0:
            pairs = self.build_level_pairs(level)
            if pairs:
                return level, pairs
            print(f"[WL-SwAV] Warning: no valid pairs at level {level}, falling back to {level - 1}")
            level -= 1
        raise RuntimeError("WL-SwAV could not build any valid positive pairs for any level.")

    def _cross_cluster_similarity(self, Z: torch.Tensor, level: int, samples: int) -> float:
        N = Z.size(0)
        if N < 2:
            return 0.0

        nodes = self.wl_engine.nodes
        attempts = 0
        success = 0
        sim_sum = 0.0
        max_attempts = max(samples * 10, samples + 1)

        while success < samples and attempts < max_attempts:
            i = random.randrange(N)
            j = random.randrange(N)
            if i == j:
                attempts += 1
                continue
            node_i = nodes[i]
            node_j = nodes[j]
            cid_i = self.wl_engine.get_cluster_id(node_i, level)
            cid_j = self.wl_engine.get_cluster_id(node_j, level)
            if cid_i is None or cid_j is None or cid_i == cid_j:
                attempts += 1
                continue
            sim_sum += (Z[i] @ Z[j]).item()
            success += 1
            attempts += 1

        return sim_sum / success if success > 0 else 0.0

    def _log_raw_losses(
        self,
        epoch: int,
        swav_loss: torch.Tensor,
        rep_loss: torch.Tensor,
        feat_loss: torch.Tensor,
    ) -> None:
        if epoch != 0:
            return
        raw_s = swav_loss.item()
        raw_r = rep_loss.item()
        raw_f = feat_loss.item()
        print(
            f"[WL-SwAV | LOSS SCALE] Raw L_SwAV={raw_s:.4f}  "
            f"Raw L_repulse={raw_r:.4f}  Raw L_feat={raw_f:.4f}"
        )

    def _warn_loss_scale(
        self,
        epoch: int,
        swav_loss: torch.Tensor,
        rep_loss: torch.Tensor,
        feat_loss: torch.Tensor,
    ) -> None:
        if epoch < 9:
            return
        raw_s = swav_loss.item()
        if abs(raw_s) < 1e-8:
            return
        raw_r = rep_loss.item()
        raw_f = feat_loss.item()
        ratio_repulse = (
            abs(self.loss_fn.lambda_uniform * raw_r) / abs(raw_s)
            if abs(raw_r) > 0
            else 0.0
        )
        ratio_feat = (
            abs(self.loss_fn.lambda_feat * raw_f) / abs(raw_s)
            if abs(raw_f) > 0
            else 0.0
        )
        if not (0.05 < ratio_repulse < 20):
            suggestion = (
                max(1e-6, abs(raw_s) / (abs(raw_r) + 1e-9))
                if abs(raw_r) > 0
                else 1e-6
            )
            logger.warning(
                "[WL-SwAV | SCALE] repulse ratio="
                f"{ratio_repulse:.2f} — consider lambda_uniform≈{suggestion:.5f}"
            )
        if not (0.05 < ratio_feat < 20):
            suggestion = (
                max(1e-6, abs(raw_s) / (abs(raw_f) + 1e-9))
                if abs(raw_f) > 0
                else 1e-6
            )
            logger.warning(
                "[WL-SwAV | SCALE] feat ratio="
                f"{ratio_feat:.2f} — consider lambda_feat≈{suggestion:.5f}"
            )

    def train_epoch(
        self, epoch: int, data: Data
    ) -> Tuple[float, float, float, float, torch.Tensor, int, int]:
        self.model.train()
        requested_level = self.get_current_level(epoch)
        used_level, pairs = self._select_level_with_pairs(requested_level)

        if epoch == 0:
            print(
                f"[WL-SwAV] Epoch 1 feature mean abs: {data.x.abs().mean().item():.6f}"
            )

        with torch.enable_grad():
            Z = self.model(data.x, data.edge_index)

        random.shuffle(pairs)
        batch_size = getattr(self.config, "batch_size", 512)
        num_batches = math.ceil(len(pairs) / batch_size)
        batch_losses = []

        for batch_idx in range(num_batches):
            chunk = pairs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            if not chunk:
                continue

            v_idx = torch.tensor([p[0] for p in chunk], dtype=torch.long, device=self.device)
            u_idx = torch.tensor([p[1] for p in chunk], dtype=torch.long, device=self.device)
            z_v = Z[v_idx]
            z_u = Z[u_idx]

            loss, q_v = self.loss_fn(z_v, z_u)
            batch_losses.append(loss)
            self._last_q_v = q_v.detach()

        if batch_losses:
            swav_loss = torch.stack(batch_losses).mean()
        else:
            swav_loss = torch.tensor(0.0, device=Z.device)

        l_repulse = self.loss_fn.repulsion_loss(Z, used_level, self.wl_engine)
        l_feat = self.loss_fn.feature_loss(Z, data.x)
        self._log_raw_losses(epoch, swav_loss, l_repulse, l_feat)
        self._warn_loss_scale(epoch, swav_loss, l_repulse, l_feat)
        total_loss = (
            swav_loss +
            self.loss_fn.lambda_uniform * l_repulse
            + self.loss_fn.lambda_feat * l_feat
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(list(self.model.parameters()) + list(self.loss_fn.parameters()), 1.0)

        if epoch < getattr(self.config, "freeze_prototypes_epochs", 0) and self.loss_fn.prototypes is not None:
            self.loss_fn.prototypes.grad = None

        self.optimizer.step()
        self.loss_fn.normalize_prototypes()

        return (
            total_loss.item(),
            swav_loss.item(),
            l_repulse.item(),
            l_feat.item(),
            Z.detach(),
            requested_level,
            used_level,
        )

    def compute_ablation_stats(self, Z: torch.Tensor, requested_level: int, used_level: int) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"levels": {}, "requested_level": requested_level, "used_level": used_level}
        level_stats: Dict[int, Dict[str, float]] = {}

        for t in range(1, requested_level + 1):
            clusters = self.wl_engine.level_nodes.get(t, [])
            intra_sum = 0.0
            intra_count = 0
            util = 0

            for cid in clusters:
                members = self.wl_engine.tree_members.get(cid, [])
                if len(members) < 2:
                    continue
                util += 1

                idx = torch.tensor(
                    [self.wl_engine.node2idx[n] for n in members],
                    device=Z.device,
                    dtype=torch.long,
                )
                emb = Z[idx]
                dot = emb @ emb.t()
                n = emb.size(0)
                pair_sum = torch.triu(dot, diagonal=1).sum().item()
                pair_count = n * (n - 1) / 2
                intra_sum += pair_sum
                intra_count += pair_count

            s_intra = intra_sum / intra_count if intra_count > 0 else 0.0
            s_inter = self._cross_cluster_similarity(Z, t, samples=min(1000, Z.size(0)))
            d_ratio = s_intra / (s_inter + 1e-8)

            level_stats[t] = {
                "S_intra": s_intra,
                "S_inter": s_inter,
                "D": d_ratio,
                "U": util,
            }

        stats["levels"] = level_stats

        if self._last_q_v is not None and self._last_q_v.numel() > 0:
            entropy = - (self._last_q_v * torch.log(self._last_q_v + 1e-8)).sum(dim=1)
            H = entropy.mean().item()
        else:
            H = 0.0

        stats["H"] = H
        kappa = max(1, getattr(self.loss_fn, "K", 1))
        threshold = 0.1 * math.log(kappa)
        if H < threshold:
            print(f"[WL-SwAV] Warning: soft-code entropy {H:.4f} below {threshold:.4f} (possible prototype collapse).")

        D_tstar = 0.0
        if used_level in level_stats:
            st = level_stats[used_level]
            D_tstar = st["D"]
        stats["D_tstar"] = D_tstar

        return stats

    def train(self, data: Data, dataset: Any, n_epochs: int, log_every: int) -> Dict[str, Any]:
        best_acc = 0.0
        best_state: Optional[Dict[str, torch.Tensor]] = None
        final_stats: Dict[str, Any] = {}

        for epoch in range(n_epochs):
            (
                total_loss,
                swav_loss,
                rep_loss,
                feat_loss,
                Z,
                requested_level,
                used_level,
            ) = self.train_epoch(epoch, data)
            if self.scheduler is not None:
                self.scheduler.step()

            if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
                stats = self.compute_ablation_stats(Z, requested_level, used_level)
                final_stats = stats
                acc = evaluate_linear_probe(
                    self.model,
                    data,
                    dataset.num_classes,
                    self.device,
                )
                if acc > best_acc:
                    best_acc = acc
                    best_state = copy.deepcopy(self.model.state_dict())

                level_summary = ", ".join(
                    f"t={t}: S_intra={vals['S_intra']:.4f}, S_inter={vals['S_inter']:.4f}, U={vals['U']}"
                    for t, vals in sorted(stats["levels"].items())
                )
                print(
                    f"[WL-SwAV | {getattr(dataset, 'name', 'dataset'):<12}] "
                    f"Epoch {epoch + 1:03d}/{n_epochs}  "
                    f"Loss: {total_loss:.4f}  "
                    f"(L_SwAV={swav_loss:.4f}  L_repulse={rep_loss:.4f}  L_feat={feat_loss:.4f})  "
                    f"Level(req={requested_level}, used={used_level})  "
                    f"H={stats['H']:.4f}  D_t*={stats['D_tstar']:.4f}  "
                    f"{level_summary}  Acc={acc:.4f}"
                )

        return {
            "dataset": getattr(dataset, "name", str(dataset)),
            "best_accuracy": best_acc,
            "epochs": n_epochs,
            "final_stats": final_stats,
            "best_state": best_state,
        }
