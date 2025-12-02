# src/data/datasets.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import (
    Planetoid,
    Actor,
    WikipediaNetwork,
)


DatasetName = Literal[
    "Cora",
    "Citeseer",
    "Pubmed",
    "Actor",
    "Squirrel",
]


@dataclass
class NodeDataset:
    data: Data
    name: str
    num_features: int
    num_classes: int



def normalize_features_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Row-wise L1 normalization (standard in citation-style datasets).
    """
    row_sum = x.sum(dim=1, keepdim=True)
    row_sum[row_sum == 0.0] = 1.0
    return x / row_sum



def load_dataset(
    name: DatasetName,
    root: str = "./data",
    normalize_features: bool = True,
) -> NodeDataset:
    """
    Load standard node classification datasets.

    Supported:
      - Planetoid: Cora, Citeseer, Pubmed
      - Actor
      - Squirrel
    """


    if name in ("Cora", "Citeseer", "Pubmed"):

        dataset = Planetoid(
                    root=root, 
                    name=name
                 )
        data = dataset[0]

    elif name == "Actor":

        dataset = Actor(
                    root=root
                )
        data = dataset[0]

    elif name == "Squirrel":

        dataset = WikipediaNetwork(
            root=root,
            name="squirrel",
            geom_gcn_preprocess=True,
        )
        data = dataset[0]

    else:
        raise ValueError(f"Unknown dataset: {name}")


    if normalize_features and data.x is not None:
        data.x = normalize_features_fn(data.x)


    assert data.x is not None, f"{name}: missing node features."
    assert data.y is not None, f"{name}: missing node labels."
    assert data.edge_index is not None, f"{name}: missing edge_index."

    for mask in ["train_mask", "val_mask", "test_mask"]:
        if getattr(data, mask, None) is None:
            raise ValueError(f"{name}: missing {mask}")


    num_features = data.x.size(-1)
    num_classes = int(data.y.max().item()) + 1

    return NodeDataset(
        data=data,
        name=name,
        num_features=num_features,
        num_classes=num_classes,
    )


def get_splits(data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return train / val / test boolean masks.
    """
    return data.train_mask, data.val_mask, data.test_mask
