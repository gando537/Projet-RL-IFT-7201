import envs
import numpy as np
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnasium.envs.registration import register

class DangerousPendulumEnv(PendulumEnv):
    def __init__(self, render_mode=None, g=15.0):
        super().__init__(render_mode=render_mode)
        self.max_torque = 0.5     # moteur très limité
        self.max_speed = 12
        # self.g = 15.0             # gravité plus forte
        self.g = g
        

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # On introduit une condition dangereuse : si l'angle dépasse 160 degrés
        theta = np.arctan2(obs[1], obs[0])  # angle réel
        danger = abs(theta) > (160 * np.pi / 180)

        info["is_dangerous"] = danger
        if danger:
            reward -= 10  # pénalité forte si l’angle est trop extrême

        return obs, reward, terminated, truncated, info

register(
    id="PendulumDangerous-v1",
    entry_point="envs.dangerous_pendulum:DangerousPendulumEnv",
)

