
# Force l'enregistrement dès l'import du package `envs`
from gymnasium.envs.registration import register
from .shielding_wrapper import ShieldingWrapper
from .dangerous_pendulum import DangerousPendulumEnv
