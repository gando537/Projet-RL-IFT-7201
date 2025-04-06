import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    # Crée l'environnement
    env = gym.make("CartPole-v1")

    # Crée l'agent DQN
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, buffer_size=50000, exploration_fraction=0.1)

    # Entraîne le modèle
    model.learn(total_timesteps=100_000)

    # Sauvegarde du modèle
    model.save("models/dqn_cartpole")

    # Évalue la politique entraînée
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Récompense moyenne sur 10 épisodes : {mean_reward:.2f} ± {std_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
