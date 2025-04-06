import gymnasium as gym
import envs.simulation_env  # importe l’enregistrement de l'env personnalisé
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    # Charge le modèle entraîné sur l'environnement de simulation
    model = DQN.load("models/dqn_cartpole")

    # Crée l'environnement "réel" (modifié)
    env = gym.make("CartPoleReal-v1", render_mode="human")

    # Évalue la politique dans le nouvel environnement
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)

    print(f"Récompense moyenne (réalité modifiée) : {mean_reward:.2f} ± {std_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
