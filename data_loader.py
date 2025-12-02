import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WebKB, Actor, Amazon, WikipediaNetwork
from torch_geometric.utils import to_undirected

def get_dataset(name, root="./data"):
    """
    Charge et standardise les datasets (Homophiles & Hétérophiles).
    """
    name = name.lower()
    
    # --- 1. CITATION (Homophiles) ---
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=root, name=name.capitalize(), transform=T.NormalizeFeatures())
        data = dataset[0]

    # --- 2. AMAZON (Homophile, Large) ---
    elif name == 'amazon-photo':
        dataset = Amazon(root=root, name='Photo', transform=T.NormalizeFeatures())
        data = dataset[0]
        # Pas de split officiel -> On génère un Random Split standard
        # (20 par classe en train, 30 en val, le reste en test)
        split = T.RandomNodeSplit(num_val=30, num_test=0, num_train_per_class=20)
        data = split(data)

    # --- 3. WEBKB (Hétérophiles, Petits) ---
    elif name in ['texas', 'wisconsin', 'cornell']:
        dataset = WebKB(root=root, name=name.capitalize(), transform=T.NormalizeFeatures())
        data = dataset[0]
        # Ces datasets ont 10 masques [N, 10]. On prend le split 0.
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]

    # --- 4. ACTOR (Hétérophile, Moyen) ---
    elif name == 'actor':
        dataset = Actor(root=root, transform=T.NormalizeFeatures())
        data = dataset[0]
        # Idem, 10 splits, on prend le 0
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]

    # --- 5. WIKIPEDIA (Hétérophiles, Denses) ---
    elif name in ['squirrel', 'chameleon']:
        # IMPORTANT: geom_gcn_preprocess=True pour avoir la version utilisée dans les papiers SOTA
        dataset = WikipediaNetwork(root=root, name=name.capitalize(), geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        data = dataset[0]
        # Idem, 10 splits, on prend le 0
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]

    else:
        raise ValueError(f"Dataset '{name}' inconnu. Options: cora, citeseer, amazon-photo, actor, squirrel, chameleon, texas, wisconsin")

    # Nettoyage : Toujours travailler en non-dirigé pour WL et GIN
    if not data.is_undirected():
        data.edge_index = to_undirected(data.edge_index)

    return data, dataset.num_classes