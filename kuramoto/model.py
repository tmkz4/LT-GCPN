import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GridKuramotoPolicy(nn.Module):
    """
    Réseau Actor-Critic pour le Traveller sur grille spatiale 2D.
    Exploite les offsets géométriques fixes (2*M actions + 1 noop).
    """
    def __init__(self, num_node_features=8, hidden_dim=64):
        super(GridKuramotoPolicy, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # MLPs pour évaluer les actions basées sur les embeddings du nœud courant (u) et de la cible (v) + offset
        input_dim = hidden_dim * 2 + 2  # h_u + h_v + (dx, dy)
        
        self.toggle_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.move_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.noop_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.critic_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        agent_pos = data.agent_pos
        cand = data.cand          # Nœuds cibles atteints par chaque offset
        offsets = data.offsets    # Coordonnées (dx, dy) de chaque offset
        mask = data.mask          # Masque booléen des actions valides
        
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        
        num_nodes = x.size(0)
        M = offsets.size(0)
        h_u = h[agent_pos]
        
        logits = []
        
        # 1. Actions Toggle (0 à M-1)
        for k in range(M):
            v = cand[k].item()
            offset_k = offsets[k]
            if v >= 0:
                feat = torch.cat([h_u, h[v], offset_k], dim=-1)
            else:
                feat = torch.cat([h_u, torch.zeros_like(h_u), offset_k], dim=-1)
            logits.append(self.toggle_mlp(feat).squeeze())
            
        # 2. Actions Move (M à 2M-1)
        for k in range(M):
            v = cand[k].item()
            offset_k = offsets[k]
            if v >= 0:
                feat = torch.cat([h_u, h[v], offset_k], dim=-1)
            else:
                feat = torch.cat([h_u, torch.zeros_like(h_u), offset_k], dim=-1)
            logits.append(self.move_mlp(feat).squeeze())
            
        # 3. Action Noop (2M)
        graph_repr = global_mean_pool(h, batch=torch.zeros(num_nodes, dtype=torch.long, device=x.device))
        logits.append(self.noop_mlp(graph_repr).squeeze())
        
        full_logits = torch.stack(logits)
        
        # Application stricte du masque d'action (les actions hors grille ou invalides sont pénalisées par -inf)
        if mask is not None:
            full_logits = torch.where(mask, full_logits, torch.tensor(-1e9, device=full_logits.device))
            
        state_value = self.critic_mlp(graph_repr).squeeze()
        
        return full_logits, state_value