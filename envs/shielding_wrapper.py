import numpy as np
from gymnasium import Wrapper

class ShieldingWrapper(Wrapper):
    def __init__(self, env, angle_threshold=np.radians(130)):
        super().__init__(env)
        self.angle_threshold = angle_threshold
        self.shield_count = 0
        self.total_steps = 0
        self.shield_log = []

    def reset(self, **kwargs):
        self.shield_count = 0
        self.shield_log = []
        self.total_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.total_steps += 1  # Compte le nombre total d'actions proposées

        # On récupère l'état actuel via l'observation précédente
        # Gymnasium donne self.env.unwrapped.state, mais on ne doit pas y toucher directement
        # On va estimer l'angle en fonction de l'observation envoyée par step

        # Appelle l'environnement sans modifier l'action (encore)
        obs, reward, terminated, truncated, info = self.env.step(action)

        # On recalcule l'angle à partir de l'observation retournée
        theta = np.arctan2(obs[1], obs[0])

        if abs(theta) > self.angle_threshold:
            print(f"[SHIELDING] Action bloquée car angle = {np.degrees(theta):.1f}°")
            self.shield_count += 1
            self.shield_log.append({
                "angle_deg": np.degrees(theta),
                "original_action": float(action[0])
            })
            # On revient au même état mais avec action nulle
            obs, reward, terminated, truncated, info = self.env.step(np.array([0.0]))

        return obs, reward, terminated, truncated, info

    def safety_score(self):
        if self.total_steps == 0:
            return 100.0
        return 100 * (1 - self.shield_count / self.total_steps)
