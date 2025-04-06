import gymnasium as gym
import envs.simulation_env  # important pour que l'environnement soit bien enregistré

env = gym.make("CartPoleReal-v1")

