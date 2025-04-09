import envs
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnasium.envs.registration import register

class RealisticPendulumEnv(PendulumEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.max_speed = 10
        self.max_torque = 2.5
        self.g = 12.0

register(
    id="PendulumReal-v1",
    entry_point="envs.realistic_pendulum:RealisticPendulumEnv",
)
