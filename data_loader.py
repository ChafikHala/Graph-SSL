import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WebKB, Actor
from torch_geometric.utils import to_undirected

def get_dataset(name, root="./data"):
    """
    Charge automatiquement un dataset (Homophile ou Hétérophile).
    """
    name = name.lower()
    
    # 1. Datasets de citation (Homophiles)
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=root, name=name.capitalize(), transform=T.NormalizeFeatures())
        data = dataset[0]
        
    # 2. Datasets WebKB (Hétérophiles : Texas, Wisconsin, Cornell)
    elif name in ['texas', 'wisconsin', 'cornell']:
        dataset = WebKB(root=root, name=name.capitalize(), transform=T.NormalizeFeatures())
        data = dataset[0]
        # Ces datasets ont souvent 10 splits pré-calculés. On prend le premier par défaut.
        # data.train_mask est une matrice [N, 10], on prend la colonne 0
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]
        
    # 3. Actor (Hétérophile)
    elif name == 'actor':
        dataset = Actor(root=root, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]
        
    else:
        raise ValueError(f"Dataset {name} non supporté.")

    # Nettoyage du graphe (Undirected est mieux pour WL)
    if not data.is_undirected():
        data.edge_index = to_undirected(data.edge_index)

    return data, dataset.num_classes