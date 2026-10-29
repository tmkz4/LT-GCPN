import os
import matplotlib
matplotlib.use('Agg')

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from tqdm import tqdm
from environment import GridKuramotoRepairEnv, EnvConfig
from model import GridKuramotoPolicy

def train_ppo_grid(num_episodes=200, lr=0.0003, gamma=0.99, ppo_epochs=4, clip_eps=0.2):
    checkpoint_dir = "checkpoints_grid"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # Configuration de la grille (ex: 8x8 = 64 nœuds)
    cfg = EnvConfig(n=4, radius=1.0, lesion_frac=0.05, lesion_mode="random", reward_mode="mixed", max_steps=16)
    env = GridKuramotoRepairEnv(cfg)
    model = GridKuramotoPolicy(num_node_features=8, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    episode_rewards = []
    episode_successes = []
    
    entropy_start = 0.05
    entropy_end = 0.01
    best_success_rate = -1.0
    
    print(f"\n--- Entraînement PPO GTCPN (Grille {cfg.n}x{cfg.n} = {env.N} nœuds) ---")
    
    pbar = tqdm(range(1, num_episodes + 1), desc="Entraînement Grille Spatiale")
    
    for episode in pbar:
        progress = (episode - 1) / max(1, num_episodes - 1)
        entropy_coef = entropy_start * (1 - progress) + entropy_end * progress
        
        state = env.reset()
        done = False
        episode_reward = 0
        is_healed = False
        
        states = []
        actions = []
        old_log_probs = []
        values = []
        rewards = []
        
        while not done:
            with torch.no_grad():
                action_logits, state_value = model(state)
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
            next_state, reward, done, info = env.step(action.item())
            
            episode_reward += reward
            is_healed = info.get("is_healed", False)
            
            states.append(state)
            actions.append(action)
            old_log_probs.append(log_prob)
            values.append(state_value)
            rewards.append(reward)
            
            state = next_state
            
            if info.get("truncated", False):
                break
                
        # --- Optimisation PPO ---
        if len(rewards) > 0:
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
                
            returns = torch.tensor(returns, dtype=torch.float)
            values = torch.stack([v.detach().view(-1) for v in values]).view(-1)
            
            advantages = returns - values
            if len(advantages) > 1 and advantages.std() > 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                
            actions = torch.stack(actions)
            old_log_probs = torch.stack(old_log_probs)
            
            for _ in range(ppo_epochs):
                for i in range(len(rewards)):
                    state_i = states[i]
                    logits, val = model(state_i)
                    dist = Categorical(logits=logits)
                    new_log_prob = dist.log_prob(actions[i])
                    entropy = dist.entropy()
                    
                    ratio = torch.exp(new_log_prob - old_log_probs[i].detach())
                    adv = advantages[i]
                    
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
                    policy_loss = -torch.min(surr1, surr2) - entropy_coef * entropy
                    
                    value_loss = (val - returns[i]) ** 2
                    loss = policy_loss + 0.5 * value_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        episode_rewards.append(episode_reward)
        episode_successes.append(1 if is_healed else 0)
        
        if episode >= 50:
            avg_reward = np.mean(episode_rewards[-50:])
            success_rate = np.mean(episode_successes[-50:]) * 100
            
            pbar.set_postfix({
                "Reward (50)": f"{avg_reward:.1f}",
                "Succès (50)": f"{success_rate:.1f}%",
                "Entropie": f"{entropy_coef:.4f}"
            })
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_grid_model.pth"))

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final_grid_model.pth"))
    print(f"\nEntraînement terminé. Meilleur taux de succès : {best_success_rate:.1f}%")

    # --- Génération et sauvegarde du graphique des courbes d'entraînement ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Moyenne mobile sur 10 épisodes pour lisser les courbes
    window = min(10, len(episode_rewards))
    if window > 0:
        smoothed_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        smoothed_successes = np.convolve(episode_successes, np.ones(window)/window, mode='valid') * 100
        x_smooth = range(window, len(episode_rewards) + 1)
    else:
        smoothed_rewards = episode_rewards
        smoothed_successes = [s * 100 for s in episode_successes]
        x_smooth = range(1, len(episode_rewards) + 1)

    # 1. Courbe des récompenses
    ax1.plot(range(1, len(episode_rewards) + 1), episode_rewards, color='lightblue', alpha=0.4, label="Récompense brute")
    if window > 0:
        ax1.plot(x_smooth, smoothed_rewards, color='royalblue', linewidth=2, label=f"Moyenne mobile ({window} ép.)")
    ax1.set_ylabel("Récompense cumulée")
    ax1.set_title("Évolution des performances d'entraînement (GTCPN PPO - Grille 8x8)")
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 2. Courbe du taux de succès
    success_pct = [s * 100 for s in episode_successes]
    ax2.plot(range(1, len(success_pct) + 1), success_pct, color='lightgreen', alpha=0.3, label="Succès brut")
    if window > 0:
        ax2.plot(x_smooth, smoothed_successes, color='forestgreen', linewidth=2, label=f"Taux de succès glissant ({window} ép.)")
    ax2.set_xlabel("Épisodes")
    ax2.set_ylabel("Taux de succès (%)")
    ax2.set_ylim(-5, 105)
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plot_path = "visualizations/training_curves.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Courbes d'entraînement sauvegardées avec succès dans : {plot_path}")

if __name__ == "__main__":
    train_ppo_grid()