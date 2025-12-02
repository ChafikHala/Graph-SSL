import torch
import torch.nn as nn
import torch.nn.functional as F

class ExtendedMoCHILoss(nn.Module):
    def __init__(self, temperature=0.1, num_negatives=128):
        """
        Extended InfoNCE Loss with MoCHi hard negative synthesis.
        
        Args:
            temperature (float): Scaling factor for logits.
            num_negatives (int): Number of synthetic negatives to generate (size of Q in MoCHi).
        """
        super(ExtendedMoCHILoss, self).__init__()
        self.tau = temperature
        self.num_negatives = num_negatives

    def forward(self, anchor, positives, hard_negatives):
        """
        Computes the loss for a SINGLE anchor node.
        
        Args:
            anchor (Tensor): Embedding of the anchor node [1, D].
            positives (Tensor): Embeddings of all positive nodes (view2 match + intersection) [N_pos, D].
            hard_negatives (Tensor): Embeddings of hard negative nodes found by DualViewMiner [N_hard, D].
        """
        # 1. Normalize everything
        anchor = F.normalize(anchor, dim=1)
        positives = F.normalize(positives, dim=1)
        
        # If no hard negatives exist, fall back to standard negatives (randomly sampled from batch if needed)
        # But here we assume hard_negatives are passed.
        if hard_negatives.size(0) > 0:
            hard_negatives = F.normalize(hard_negatives, dim=1)
            
            # 2. MoCHi Synthesis (Generating Synthetic Hard Negatives)
            # Create "Harder" and "Hardest" negatives by mixing embeddings
            synthetic_negs = self.mochi_generation(anchor, hard_negatives)
            
            # Combine real hard negatives and synthetic ones
            # Total negatives = {k-} U {h-} U {h'-}
            all_negatives = torch.cat([hard_negatives, synthetic_negs], dim=0)
        else:
            # Fallback if intersection logic found no hard negatives for this node
            # Use random noise or just skip synthesis (implementation choice)
            all_negatives = hard_negatives

        # 3. Calculate Logits
        # Positive logits: s(q, k+) / tau
        # Shape: [N_pos]
        pos_logits = torch.matmul(positives, anchor.t()).squeeze(-1) / self.tau
        
        # Negative logits: s(q, negs) / tau
        # Shape: [Total_Negs]
        if all_negatives.size(0) > 0:
            neg_logits = torch.matmul(all_negatives, anchor.t()).squeeze(-1) / self.tau
        else:
            # Theoretical edge case: no negatives at all (unlikely in batch training)
            neg_logits = torch.tensor([-1e9]).to(anchor.device)

        # 4. Compute Extended InfoNCE
        # Formula: - log [ sum(exp(pos)) / (sum(exp(pos)) + sum(exp(neg))) ]
        # Since we have multiple positives, we average the log-prob over them.
        
        # LogSumExp of negatives (denominator part 2)
        # MoCHi(q) in equation (4) corresponds to sum(exp(neg_logits))
        neg_sum_exp = torch.sum(torch.exp(neg_logits))
        
        loss = 0
        n_pos = pos_logits.size(0)
        
        if n_pos > 0:
            # We iterate because the formula sums over {k+}
            for i in range(n_pos):
                pos_val = torch.exp(pos_logits[i])
                # L = - log ( exp(pos_i) / (exp(pos_i) + sum_exp(negs)) )
                denom = pos_val + neg_sum_exp
                loss += -torch.log(pos_val / (denom + 1e-8)) # epsilon for stability
            
            # Average over number of positives
            loss = loss / n_pos
            
        return loss

    def mochi_generation(self, anchor, hard_negs):
        """
        Synthesizes harder negatives by interpolating features.
        MoCHi Logic:
        1. {h'}: Mix of Hard Negatives + Anchor (Hardest)
        2. {h}: Mix of Hard Negatives + Hard Negatives (Harder)
        """
        N = hard_negs.size(0)
        if N < 2: return hard_negs # Not enough to mix
        
        # How many synthetic samples to make? Let's aim for self.num_negatives
        # We split budget 50/50 between mixing with anchor and mixing with other negs
        num_mix_anchor = self.num_negatives // 2
        num_mix_negs = self.num_negatives - num_mix_anchor
        
        # --- Type 1: Hardest {h'-} (Mix Anchor + Negs) ---
        # Select random hard negs to mix with anchor
        indices = torch.randint(0, N, (num_mix_anchor,)).to(anchor.device)
        selected_negs = hard_negs[indices]
        
        # Mixing coefficient alpha ~ U(0.1, 0.5) to stay "negative" but get closer to anchor
        alpha = torch.rand(num_mix_anchor, 1).to(anchor.device) * 0.4 + 0.1 
        
        hardest_negs = (1 - alpha) * selected_negs + alpha * anchor
        hardest_negs = F.normalize(hardest_negs, dim=1) # Re-normalize
        
        # --- Type 2: Harder {h-} (Mix Negs + Negs) ---
        # Select pairs of random hard negs
        idx_a = torch.randint(0, N, (num_mix_negs,)).to(anchor.device)
        idx_b = torch.randint(0, N, (num_mix_negs,)).to(anchor.device)
        
        negs_a = hard_negs[idx_a]
        negs_b = hard_negs[idx_b]
        
        beta = torch.rand(num_mix_negs, 1).to(anchor.device) * 0.4 + 0.3 # Mix between 0.3 and 0.7
        
        harder_negs = beta * negs_a + (1 - beta) * negs_b
        harder_negs = F.normalize(harder_negs, dim=1)
        
        return torch.cat([hardest_negs, harder_negs], dim=0)