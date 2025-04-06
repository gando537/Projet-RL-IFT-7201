import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.envs.registration import register

class RealisticCartPoleEnv(CartPoleEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        # On modifie la physique du système pour simuler une "réalité" différente
        self.gravity = 12.0          # par défaut : 9.8
        self.masscart = 1.5          # par défaut : 1.0
        self.masspole = 0.2          # par défaut : 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.7            # demi-longueur du pendule (par défaut : 0.5)
        self.polemass_length = self.masspole * self.length

# Enregistre l’environnement modifié sous un nouveau nom
register(
    id='CartPoleReal-v1',
    entry_point='envs.simulation_env:RealisticCartPoleEnv',
)
