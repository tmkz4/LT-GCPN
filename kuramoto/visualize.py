import os
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.distributions import Categorical

from environment import GridKuramotoRepairEnv, EnvConfig
from model import GridKuramotoPolicy

def visualize_episode(checkpoint_path="checkpoints_grid/best_grid_model.pth", use_random=False):
    # Configuration de l'environnement
    cfg = EnvConfig(n=8, radius=1.0, reward_mode="mixed", max_steps=60)
    env = GridKuramotoRepairEnv(cfg)
    
    if not use_random:
        # Chargement du modèle entraîné
        model = GridKuramotoPolicy(num_node_features=8, hidden_dim=64)
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"Modèle chargé avec succès depuis : {checkpoint_path}")
        else:
            print("⚠️ Aucun checkpoint trouvé. Utilisation d'un modèle non entraîné (poids aléatoires).")
        model.eval()
    else:
        print("🎲 Mode ALÉATOIRE activé : le Traveller effectue des choix aléatoires parmi les actions valides.")
    
    state = env.reset()
    
    # Capture des lésions initiales
    initial_lesions = env.target_edges - env.edges
    done = False
    
    trajectory = [env.agent]
    actions_taken = []
    spreads = [env.spread]
    
    print(f"\n--- Début de l'épisode de visualisation ({'Aléatoire' if use_random else 'Modèle GTCPN'}) ---")
    print(f"Nombre de liaisons lésées au départ : {len(initial_lesions)}")
    
    step_count = 0
    while not done:
        mask = state.mask if hasattr(state, 'mask') else env.action_mask()
        
        if use_random:
            # Sélection aléatoire parmi les actions valides (masque True) sans passer par .numpy()
            if isinstance(mask, torch.Tensor):
                valid_actions = torch.nonzero(mask).squeeze(-1).tolist()
                # Si mask est un scalaire ou un tenseur 0D transformé en liste vide/int unique
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]
            else:
                valid_actions = np.where(mask)[0].tolist()
            
            action = int(env.rng.choice(valid_actions))
        else:
            with torch.no_grad():
                logits, _ = model(state)
                if state.mask is not None:
                    logits = torch.where(state.mask, logits, torch.tensor(-1e9, device=logits.device))
                dist = Categorical(logits=logits)
                action = dist.sample().item()
            
        kind, k = env._decode(action)
        target_node = env._neighbor(env.agent, k) if kind != "noop" else -1
        
        state, reward, done, info = env.step(action)
        
        trajectory.append(env.agent)
        actions_taken.append((kind, k, target_node))
        spreads.append(env.spread)
        step_count += 1
        
        print(f"Étape {step_count:2d} | Action : {kind.upper():6s} (offset slot: {k}, cible: {target_node:2d}) | Spread : {env.spread:.4f}")
        
        if info.get("truncated", False):
            break

    success = info.get("is_healed", False)
    print(f"\nÉpisode terminé en {step_count} étapes.")
    print(f"Statut de guérison (Succès) : {success}")
    
    # --- Génération des graphiques ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    policy_name = "Aléatoire (Random Baseline)" if use_random else "GTCPN (Modèle Entraîné)"
    color_traj = 'darkorange' if use_random else 'dodgerblue'
    color_spread = 'darkorange' if use_random else 'royalblue'
    
    # 1. Trajectoire et lésions
    ax1.set_title(f"Trajectoire - {policy_name} (Grille {cfg.n}x{cfg.n})\nSuccès: {success} en {step_count} étapes", fontsize=12)
    ax1.set_xlim(-0.5, cfg.n - 0.5)
    ax1.set_ylim(-0.5, cfg.n - 0.5)
    ax1.set_xticks(range(cfg.n))
    ax1.set_yticks(range(cfg.n))
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    for u, v in env.target_edges:
        p1, p2 = env.pos[u], env.pos[v]
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color='lightgray', linestyle='-', linewidth=1, zorder=1)
        
    for u, v in initial_lesions:
        p1, p2 = env.pos[u], env.pos[v]
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color='crimson', linestyle='--', linewidth=2.5, alpha=0.8, zorder=2, label='Lésion initiale' if 'Lésion initiale' not in [l.get_label() for l in ax1.get_lines()] else "")

    pos_x, pos_y = env.pos[:, 0], env.pos[:, 1]
    ax1.scatter(pos_x, pos_y, c='white', edgecolors='black', s=120, zorder=3, label="Nœuds")
    
    traj_coords = env.pos[trajectory]
    ax1.plot(traj_coords[:, 0], traj_coords[:, 1], color=color_traj, marker='o', linewidth=2.5, zorder=4, label="Trajectoire Traveller")
    ax1.scatter(traj_coords[0, 0], traj_coords[0, 1], color='forestgreen', s=220, zorder=5, label="Départ (Spawn)")
    ax1.scatter(traj_coords[-1, 0], traj_coords[-1, 1], color='purple' if use_random else 'darkorange', s=220, zorder=5, label="Position Finale")
    ax1.legend(loc='upper right', fontsize=9)
    
    # 2. Dynamique de synchronisation (Spread)
    ax2.set_title(f"Évolution de la synchronisation - {policy_name}", fontsize=12)
    ax2.plot(spreads, color=color_spread, marker='o', linewidth=2, label="Spread (ptp des vitesses)")
    ax2.axhline(y=cfg.lock_tol, color='forestgreen', linestyle='--', linewidth=2, label=f"Seuil de verrouillage ({cfg.lock_tol})")
    ax2.set_xlabel("Nombre d'étapes")
    ax2.set_ylabel("Spread (max(v) - min(v))")
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    os.makedirs("visualizations", exist_ok=True)
    save_path = "visualizations/episode_trajectory_random.png" if use_random else "visualizations/episode_trajectory.png"
    plt.savefig(save_path, dpi=300)
    print(f"Visualisation sauvegardée avec succès dans : {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true", help="Exécuter une politique aléatoire pour la baseline")
    args = parser.parse_args()
    visualize_episode(use_random=args.random)