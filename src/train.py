import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    # Utilise un environnement à actions continues
    env = gym.make("Pendulum-v1")  # SAC nécessite un espace d'action continu

    # Crée le modèle SAC
    model = SAC("MlpPolicy", env, verbose=1, learning_rate=3e-4, buffer_size=100_000, train_freq=1)

    # Entraîne l'agent
    model.learn(total_timesteps=100_000)

    # Sauvegarde le modèle
    model.save("models/sac_pendulum")

    # Évalue la performance
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Récompense moyenne sur 10 épisodes : {mean_reward:.2f} ± {std_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
