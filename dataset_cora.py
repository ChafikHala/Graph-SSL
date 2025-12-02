import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
import os

def load_cora_raw(data_dir="./cora"):
    print("--- Loading Cora Dataset from raw files ---")
    
    content_path = os.path.join(data_dir, "cora.content")
    cites_path = os.path.join(data_dir, "cora.cites")

    # 1. Chargement des Features et Labels (cora.content)
    # Format: <paper_id> <word_attributes>+ <class_label>
    # On lit tout comme des strings d'abord
    print(f"Reading {content_path}...")
    raw_data = pd.read_csv(content_path, sep='\t', header=None)
    
    # Extraction des IDs bruts
    raw_ids = raw_data[0].values
    
    # Création du mapping : ID_Brut -> Index (0 à N-1)
    id_map = {id: i for i, id in enumerate(raw_ids)}
    num_nodes = len(raw_ids)
    
    # Extraction des Features (colonnes 1 à avant-dernière)
    features = torch.tensor(raw_data.iloc[:, 1:-1].values, dtype=torch.float)
    
    # Extraction des Labels (dernière colonne)
    labels_raw = raw_data.iloc[:, -1].values
    # Encodage des labels en entiers (String -> Int)
    label_map = {label: i for i, label in enumerate(np.unique(labels_raw))}
    labels = torch.tensor([label_map[l] for l in labels_raw], dtype=torch.long)
    
    print(f"Nodes: {num_nodes}, Feature Dim: {features.shape[1]}, Classes: {len(label_map)}")

    # 2. Chargement du Graphe (cora.cites)
    # Format: <cited_paper_id> <citing_paper_id>
    print(f"Reading {cites_path}...")
    cites_data = pd.read_csv(cites_path, sep='\t', header=None)
    
    # On mappe les IDs bruts vers nos indices 0..N-1
    # Attention : Certains IDs dans cites peuvent ne pas être dans content (rare mais possible)
    # On filtre pour garder uniquement les arêtes entre noeuds existants
    src = [id_map[i] for i in cites_data[0] if i in id_map]
    dst = [id_map[i] for i in cites_data[1] if i in id_map]
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # Rendre le graphe non-dirigé (Undirected)
    # C'est standard pour Cora : si A cite B, il y a un lien entre eux.
    # On ajoute les arêtes inverses
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Supprimer les doublons éventuels
    edge_index = torch.unique(edge_index, dim=1)

    # 3. Création de l'objet Data
    data = Data(x=features, edge_index=edge_index, y=labels)
    data.num_classes = len(label_map)
    
    return data, num_nodes, features.shape[1]

def create_masks(data, val_ratio=0.1, test_ratio=0.8):
    """Crée des masques d'entrainement/validation/test aléatoires."""
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    
    val_size = int(num_nodes * val_ratio)
    test_size = int(num_nodes * test_ratio)
    train_size = num_nodes - val_size - test_size
    
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    data.train_mask[indices[:train_size]] = True
    data.val_mask[indices[train_size:train_size+val_size]] = True
    data.test_mask[indices[train_size+val_size:]] = True
    
    print(f"Split: Train {train_size}, Val {val_size}, Test {test_size}")
    return data