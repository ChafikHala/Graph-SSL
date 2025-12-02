import torch
from torch_geometric.utils import dropout_edge, dropout_node

class GraphAugmentor:
    def __init__(self, edge_drop_prob=0.2, feature_mask_prob=0.2):
        """
        Handles graph augmentations for Contrastive Learning.
        
        Args:
            edge_drop_prob (float): Probability of removing an edge.
            feature_mask_prob (float): Probability of masking a feature dimension.
        """
        self.edge_drop_prob = edge_drop_prob
        self.feature_mask_prob = feature_mask_prob

    def augment(self, x, edge_index):
        """
        Generates an augmented graph view (G').
        
        Args:
            x (Tensor): Node features [N, F]
            edge_index (Tensor): Edges [2, E]
        Returns:
            x_aug, edge_index_aug
        """
        edge_index_aug, _ = dropout_edge(edge_index, p=self.edge_drop_prob)
        
        # Feature Masking
        x_aug = self.mask_features(x)
        
        return x_aug, edge_index_aug

    def mask_features(self, x):
        """Randomly zero-out some feature dimensions."""
        if self.feature_mask_prob == 0:
            return x
            
        mask = torch.empty_like(x).bernoulli_(1 - self.feature_mask_prob).to(x.device)
        return x * mask
    
if __name__ == "__main__":
    nodes = [0, 1, 2, 3, 4, 5, 6, 7]
    edges = [(0, 1), (0, 7), (1, 2), (7, 2), (2, 3), (2, 6), (3, 4), (6, 5), (4, 5)]
    
    x = torch.ones((8, 10), dtype=torch.float) # 10 fictionzl features
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1) # Undirected
    
    augmentor = GraphAugmentor(edge_drop_prob=0.3, feature_mask_prob=0.3)
    
    # generating view1 and view2
    x1, edge_index1 = augmentor.augment(x, edge_index)
    x2, edge_index2 = augmentor.augment(x, edge_index)
    
    print(f"Original Edges: {edge_index.shape[1]}")
    print(f"View 1 Edges:   {edge_index1.shape[1]}")
    print(f"View 2 Edges:   {edge_index2.shape[1]}")
    
    print("\nmaseked features? ?")
    print(f"Original sum: {x.sum()}")
    print(f"View 1 sum:   {x1.sum()}")