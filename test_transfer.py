import gymnasium as gym
from matplotlib import pyplot as plt
import envs
import numpy as np
from stable_baselines3 import SAC
from envs.shielding_wrapper import ShieldingWrapper
import subprocess

def run_episode(env, model, max_steps=300):
    obs, _ = env.reset()
    done, total_reward = False, 0
    step_count = 0

    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1

    safety_score = env.safety_score() if hasattr(env, "safety_score") else 0.0

    # print(f"→ Blocages de sécurité : {env.shield_count}")
    print(f"→ Score de sécurité : {safety_score:.2f}%")
    return total_reward, safety_score


def plot_rewards_and_safety(rewards_with, rewards_without, safety_with, safety_without):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # --- Graphe 1 : Récompenses ---
    reward_means = [np.mean(rewards_without), np.mean(rewards_with)]
    reward_stds = [np.std(rewards_without), np.std(rewards_with)]
    axs[0].bar(['Sans Shielding', 'Avec Shielding'], reward_means, yerr=reward_stds, capsize=10,
               color=['tomato', 'seagreen'])
    axs[0].set_title("Récompense moyenne par épisode")
    axs[0].set_ylabel("Récompense cumulée")
    axs[0].grid(axis='y')

    # --- Graphe 2 : Scores de sécurité ---
    safety_means = [np.mean(safety_without), np.mean(safety_with)]
    safety_stds = [np.std(safety_without), np.std(safety_with)]
    axs[1].bar(['Sans Shielding', 'Avec Shielding'], safety_means, yerr=safety_stds, capsize=10,
               color=['tomato', 'seagreen'])
    axs[1].set_title("Score de sécurité (%)")
    axs[1].set_ylabel("Score moyen sur 5 épisodes")
    axs[1].set_ylim(0, 100)
    axs[1].grid(axis='y')

    plt.tight_layout()
    plt.show()

def make_env(shielded=False):
    env = gym.make("PendulumDangerous-v1", render_mode=None)
    if shielded:
        return ShieldingWrapper(env)
    return env

def main():
    model = SAC.load("models/sac_pendulum")
    base_env = gym.make("PendulumDangerous-v1", render_mode=None)
    env = ShieldingWrapper(base_env)

    rewards_with = []
    rewards_without = []
    safety_with = []
    safety_without = []

    print("\n--- Avec Shielding ---")
    for _ in range(5):
        env = make_env(shielded=True)
        r, s = run_episode(env, model)
        rewards_with.append(r)
        safety_with.append(s)

    print("\n--- Sans Shielding ---")
    for _ in range(5):
        env = make_env(shielded=False)
        r, s = run_episode(env, model)
        rewards_without.append(r)
        safety_without.append(s)

    plot_rewards_and_safety(rewards_with, rewards_without, safety_with, safety_without)
    print("Evaluation du transfert")


if __name__ == "__main__":
    main()

# Exécute evaluate_transfer.py juste après
subprocess.run(["python", "src/evaluate_transfer.py"])
