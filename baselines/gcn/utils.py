from enum import Enum

import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix


class Dataset(Enum):
    Cora = 0
    CiteSeer = 1
    PubMed = 2


def load_data(dataset_name: Dataset, load_dir="planetoid"):
    dataset = Planetoid(root=load_dir, name=dataset_name.name)
    data = dataset[0]  
    num_features = dataset.num_features
    num_classes = len(set(data.y.numpy()))

    return data, num_features, num_classes

