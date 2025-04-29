import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import envs  # maintenant Python peut le trouver


def evaluate_transfer(env_name, model_path, n_episodes=5):
    """Évalue le transfert sim2real avec différentes perturbations"""
    results = {}

    for g in [9.81, 11.0, 15.0]:
        print(f"\n🌍 Gravité g = {g}")
        env = DummyVecEnv([lambda: gym.make(env_name, g=g)])
        model = SAC.load(model_path)

        rewards = []
        for i in tqdm(range(n_episodes), desc=f"Épisodes pour g={g}"):
            obs = env.reset()
            done = False
            ep_reward = 0
            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, _ = env.step(action)
                ep_reward += reward
            rewards.append(ep_reward)

        results[f"g={g}"] = np.mean(rewards)


    # Création du dossier s'il n'existe pas
    os.makedirs("results", exist_ok=True)

    plt.bar(results.keys(), results.values(), color='cornflowerblue')
    plt.title("Robustesse aux changements de gravité")
    plt.ylabel("Récompense moyenne")
    plt.tight_layout()
    plt.savefig("results/gravity_transfer.png")
    plt.show()

if __name__ == "__main__":
    evaluate_transfer("PendulumDangerous-v1", "models/sac_pendulum")
