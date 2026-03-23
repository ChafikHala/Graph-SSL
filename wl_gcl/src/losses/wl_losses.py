import torch
import torch.nn.functional as F
import random



def wl_classification_loss(logits, level_targets, active_levels):
    """
    Cross-entropy over selected WL levels.
    """
    loss = 0.0
    for t in active_levels:
        y_t = level_targets[t]["y"]
        loss = loss + torch.nn.functional.cross_entropy(logits[t], y_t)
    return loss


def hierarchy_regularization(engine, z, levels, nodes=None):
    """
    WL centroid consistency regularization.
    """
    if nodes is None:
        nodes = range(z.size(0))

    reg = torch.tensor(0.0, device=z.device)

    centroid_by_level = {
        t: engine.compute_centroids(z, t) for t in levels
    }

    for t in levels[1:]:
        t_prev = t - 1
        if t_prev not in centroid_by_level:
            continue

        cent_t = centroid_by_level[t]
        cent_prev = centroid_by_level[t_prev]

        for v in nodes:
            cid_t = engine.get_cluster_id(v, t)
            cid_prev = engine.get_cluster_id(v, t_prev)

            if cid_t in cent_t and cid_prev in cent_prev:
                reg = reg + (cent_t[cid_t] - cent_prev[cid_prev]).pow(2).sum()

    return reg / len(list(nodes))

def wl_contrastive_loss(engine, z, level, temperature=0.5):
    """
    Fully vectorized WL-InfoNCE.

    z: (N, d) normalized embeddings
    """

    device = z.device
    N = z.size(0)

    # ------------------------------------------------------------
    # 1) Similarity matrix (N x N)
    # ------------------------------------------------------------
    S = torch.mm(z, z.t()) / temperature
    expS = torch.exp(S)

    # ------------------------------------------------------------
    # 2) Build WL positive mask
    # mask_pos[u,v] = 1 if v in same WL cluster at 'level'
    # ------------------------------------------------------------
    mask_pos = torch.zeros((N, N), device=device, dtype=torch.float32)

    for u in range(N):
        cluster = engine.get_cluster_at_level(u, level)
        if cluster is None:
            continue
        mask_pos[u, cluster] = 1.0

    # remove self-similarity from positives
    mask_pos.fill_diagonal_(0.0)

    # ------------------------------------------------------------
    # 3) Negative mask = everything else
    # ------------------------------------------------------------
    mask_neg = 1.0 - mask_pos

    # ------------------------------------------------------------
    # 4) InfoNCE
    # ------------------------------------------------------------
    pos_sum = (expS * mask_pos).sum(dim=1)
    neg_sum = (expS * mask_neg).sum(dim=1)

    # avoid division by zero
    eps = 1e-9
    loss = -torch.log((pos_sum + eps) / (pos_sum + neg_sum + eps))

    return loss.mean()


def hierarchical_probabilistic_loss(engine, logits, level_targets, active_levels):
    """
    Implements Hierarchical Probabilistic Consistency Loss.
    Enforces marginal agreement between predictions at level t and level t+1.
    """
    kl_loss = 0.0
    # Ensure levels are processed in hierarchical order (top-down)
    levels = sorted(active_levels)
    
    # Iterate through pairs of parent (t) and child (t_next) levels
    for i in range(len(levels) - 1):
        t = levels[i]
        t_next = levels[i+1]
        
        # 1. Get predicted probabilities for the parent level (rho^(t))
        p_t = F.softmax(logits[t], dim=-1)
        
        # 2. Get predicted probabilities for the child level (rho^(t+1))
        p_next = F.softmax(logits[t_next], dim=-1)
        
        # Retrieve mapping dictionaries: cluster_id -> matrix_index
        cid2idx_t = level_targets[t]["cid2idx"]
        cid2idx_next = level_targets[t_next]["cid2idx"]
        
        device = p_t.device
        num_classes_t = p_t.size(1)
        num_classes_next = p_next.size(1)
        
        # 3. Construct the Adjacency/Membership matrix M [num_children, num_parents]
        # M[i, j] = 1 if child cluster i belongs to parent cluster j
        M = torch.zeros((num_classes_next, num_classes_t), device=device)
        
        for cid_next, idx_next in cid2idx_next.items():
            # Find the parent ID for the current child ID using the engine's hierarchy
            parent_cid = engine.parent.get(cid_next)
            if parent_cid in cid2idx_t:
                idx_t = cid2idx_t[parent_cid]
                M[idx_next, idx_t] = 1.0
                
        # 4. Calculate the induced parent distribution (rho_tilde^(t))
        # Summing the probabilities of children to reconstruct the parent's probability
        p_tilde_t = torch.matmul(p_next, M)
        
        # Numerical stability: clamp values to avoid log(0) which results in NaN
        eps = 1e-8
        p_tilde_t = p_tilde_t.clamp(min=eps)
        p_t = p_t.clamp(min=eps)
        
        # 5. KL Divergence: KL(Predicted_Parent || Induced_Parent_from_Children)
        # In PyTorch, F.kl_div expects: input = log_probabilities, target = probabilities
        loss_kl = F.kl_div(p_tilde_t.log(), p_t, reduction='batchmean')
        
        kl_loss += loss_kl
    
    return  kl_loss
