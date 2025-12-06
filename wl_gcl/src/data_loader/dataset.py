# src/data_loader/dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import (
    Planetoid,
    Actor,
    WikipediaNetwork,
    WebKB,
    Amazon
)

DatasetName = Literal[
    "Cora",
    "Citeseer",
    "Pubmed",
    "Actor",
    "Squirrel",
    "Chameleon",
    "Texas",
    "Wisconsin",
    "Cornell",
    "Amazon-Photo"
]

@dataclass
class NodeDataset:
    data: Data
    name: str
    num_features: int
    num_classes: int

def normalize_features_fn(x: torch.Tensor) -> torch.Tensor:
    """Row-wise L1 normalization (standard in citation-style datasets)."""
    row_sum = x.sum(dim=1, keepdim=True)
    row_sum[row_sum == 0.0] = 1.0
    return x / row_sum

def load_dataset(
    name: str,
    root: str = "./data",
    normalize_features: bool = True,
) -> NodeDataset:
    """
    Load standard node classification datasets (Homophilic & Heterophilic).
    Merged logic: Uses Hala's wrapper but Farouk's robust loading logic.
    """
    # Normalize name to match case-insensitivity if needed, but let's stick to capitalized for now
    # Name mapping to handle minor differences
    original_name = name
    name_lower = name.lower()

    # --- 1. CITATION (Homophilic) ---
    if name_lower in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(root=root, name=name.capitalize())
        data = dataset[0]

    # --- 2. AMAZON (Homophile, Large) ---
    elif name_lower == 'amazon-photo':
        dataset = Amazon(root=root, name='Photo')
        data = dataset[0]
        # Generate random split if missing
        split = T.RandomNodeSplit(num_val=30, num_test=0, num_train_per_class=20)
        data = split(data)

    # --- 3. WEBKB (Heterophilic, Small) ---
    elif name_lower in ['texas', 'wisconsin', 'cornell']:
        dataset = WebKB(root=root, name=name.capitalize())
        data = dataset[0]
        # Take the first split (standard protocol)
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]

    # --- 4. ACTOR (Heterophilic) ---
    elif name_lower == 'actor':
        dataset = Actor(root=root)
        data = dataset[0]
        # Take the first split
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]

    # --- 5. WIKIPEDIA (Heterophilic, Dense) ---
    elif name_lower in ['squirrel', 'chameleon']:
        dataset = WikipediaNetwork(root=root, name=name.capitalize(), geom_gcn_preprocess=True)
        data = dataset[0]
        # Take the first split
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]

    else:
        raise ValueError(f"Unknown dataset: {name}")

    # --- POST PROCESSING (Shared Logic) ---
    
    # 1. Feature Normalization
    if normalize_features and data.x is not None:
        data.x = normalize_features_fn(data.x)

    # 2. Undirected Edges (Crucial for WL and GIN)
    if not data.is_undirected():
        data.edge_index = to_undirected(data.edge_index)

    # 3. Validation Checks
    assert data.x is not None, f"{name}: missing node features."
    assert data.y is not None, f"{name}: missing node labels."
    assert data.edge_index is not None, f"{name}: missing edge_index."

    for mask in ["train_mask", "val_mask", "test_mask"]:
        if getattr(data, mask, None) is None:
            # Fallback: if masks are missing (rare with above logic), warn or fail
             raise ValueError(f"{name}: missing {mask}")

    num_features = data.x.size(-1)
    num_classes = int(data.y.max().item()) + 1

    return NodeDataset(
        data=data,
        name=original_name,
        num_features=num_features,
        num_classes=num_classes,
    )

def get_splits(data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return train / val / test boolean masks."""
    return data.train_mask, data.val_mask, data.test_mask