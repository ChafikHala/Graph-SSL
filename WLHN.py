import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from math import sqrt

MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}

class WLHNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, tau=1.0):
        super(WLHNEncoder, self).__init__()
        self.n_layers = n_layers
        self.scaling = torch.tanh(torch.tensor(tau / 2))
        
        # 1. Projection initiale (H^0)
        self.fc0 = nn.Linear(input_dim, hidden_dim)

        # Vecteur parent virtuel pour la racine
        self.p = (-1./sqrt(hidden_dim)) * torch.ones(hidden_dim, requires_grad=False)

        # 2. Couches MPNN (GIN)
        # Le papier utilise GIN pour mettre à jour les features Euclidiennes
        lst = list()
        # Première couche
        lst.append(GINConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
                          nn.Linear(hidden_dim, hidden_dim), nn.ReLU())))
        # Couches suivantes
        for i in range(n_layers - 1):
            lst.append(GINConv(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
                           nn.Linear(hidden_dim, hidden_dim), nn.ReLU())))
        self.conv = nn.ModuleList(lst)
        self.relu = nn.ReLU()

    # --- Utilitaires Géométrie Hyperbolique ---

    def project(self, x):
        norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
        eps = BALL_EPS[x.dtype]
        maxnorm = (1 - eps)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def logmap0(self, y):
        """Map logarithmique : Hyperbolique -> Tangent (Euclidien)"""
        y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
        return y / y_norm / 1. * torch.atanh(y_norm.clamp(-1 + 1e-15, 1 - 1e-15))

    # Reflection (circle inversion)
    def isometric_transform(self, x, a):
        r2 = torch.sum(a ** 2, dim=-1, keepdim=True) - 1.
        u = x - a
        return r2 / torch.sum(u ** 2, dim=-1, keepdim=True).clamp_min(MIN_NORM) * u + a

    def reflection_center(self, mu):
        return mu / torch.sum(mu ** 2, dim=-1, keepdim=True).clamp_min(MIN_NORM)

    def reflect_at_zero(self, x, mu):
        a = self.reflection_center(mu)
        return self.isometric_transform(x, a)

    def reflect_through_zero(self, p, q, x):
        p_ = p / torch.norm(p, dim=-1, keepdim=True).clamp_min(MIN_NORM)
        q_ = q / torch.norm(q, dim=-1, keepdim=True).clamp_min(MIN_NORM)
        r = q_ - p_
        m = torch.sum(r * x, dim=-1, keepdim=True) / torch.sum(r * r, dim=-1, keepdim=True)
        return x - 2 * r * m

    # --- Forward Pass ---

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Initialisation H^(0)
        x = self.relu(self.fc0(x))
        
        # Structures pour stocker la hiérarchie
        xs = [x]
        # Racine Z^(-1) = 0
        z = [torch.zeros(1, x.size(1), device=x.device, requires_grad=False)] 
        inv = [torch.zeros(x.size(0), dtype=torch.long, device=x.device, requires_grad=False)]
        
        # --- Étape 0 : Placement initial ---
        # On groupe les nœuds identiques (même "couleur" initiale)
        with torch.no_grad():
            unique_all, inv_all = torch.unique(x, sorted=False, return_inverse=True, dim=0)
        
        unique_all_norm = unique_all / torch.norm(unique_all, dim=1).unsqueeze(1).clamp_min(MIN_NORM)
        
        # Z^(0) : Placement initial sur la sphère
        z.append(self.scaling * unique_all_norm) 
        inv.append(inv_all)

        # --- Boucle Principale (Algorithme 2) ---
        for i in range(self.n_layers):
            # 1. MPNN Update (GIN) -> H^(i)
            x = self.conv[i](x, edge_index)
            xs.append(x)

            # 2. DiffHypCon (Algorithme 1)
            with torch.no_grad():
                # Concaténation de l'historique pour identifier l'unicité dans l'arbre WL
                unique_all, inv_all, count_all = torch.unique(torch.cat(xs, dim=1), sorted=False, return_inverse=True, return_counts=True, dim=0)
            
            # Récupérer juste les features courantes pour le placement
            unique_all = unique_all[:, -x.size(1):] 
            unique_all_norm = unique_all / torch.norm(unique_all, dim=1).unsqueeze(1).clamp_min(MIN_NORM)
            
            # Calculs pour placer les enfants
            z_children = self.scaling * unique_all_norm
            
            # Mapping enfants -> parents via les indices inverses
            t = torch.zeros(unique_all.size(0), dtype=torch.long, device=x.device)
            t.scatter_add_(0, inv_all, inv[i+1]) # Mapping vers l'étape précédente
            t = torch.div(t, count_all).long()
            
            # Récupération des positions parents/courants pour la réflexion
            z_current = torch.gather(z[i+1], 0, t.unsqueeze(1).repeat(1, z[i+1].size(1)))
            
            t = torch.zeros(unique_all.size(0), dtype=torch.long, device=x.device)
            t.scatter_add_(0, inv_all, inv[i])
            t = torch.div(t, count_all).long()
            z_parent = torch.gather(z[i], 0, t.unsqueeze(1).repeat(1, z[i].size(1)))
            
            # Application des réflexions (Sarkar construction)
            z_parent = self.reflect_at_zero(z_parent, z_current)
            z_children = self.reflect_through_zero(z_parent, self.p.to(x.device), z_children)
            z_all = self.reflect_at_zero(z_children, z_current)
            
            inv.append(inv_all)
            z.append(z_all)
            
        # --- Readout pour Contrastive Learning ---
        
        # Mapping back vers tous les nœuds originaux
        hyperbolic_node_embeddings = torch.index_select(z[-1], 0, inv[-1])
        
        # Projection vers l'espace Tangent (Euclidien plat)
        tangent_node_embeddings = self.logmap0(hyperbolic_node_embeddings)
        
        # --- Remplacement natif de scatter_add ---
        # Si data.batch n'existe pas (un seul graphe), on en crée un
        if not hasattr(data, 'batch') or data.batch is None:
             batch_idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
             batch_idx = data.batch

        batch_size = int(batch_idx.max()) + 1
        graph_embeddings = torch.zeros(batch_size, tangent_node_embeddings.size(1), device=x.device)
        
        # On étend les indices pour matcher la dim cachée
        index = batch_idx.unsqueeze(1).expand(-1, tangent_node_embeddings.size(1))
        
        # Somme native PyTorch (Pooling)
        graph_embeddings.scatter_add_(0, index, tangent_node_embeddings)
        
        return graph_embeddings