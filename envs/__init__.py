from .realistic_pendulum import RealisticPendulumEnv

# Force l'enregistrement dès l'import du package `envs`
from gymnasium.envs.registration import register
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from .dangerous_pendulum import DangerousPendulumEnv
