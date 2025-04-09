import gymnasium as gym
import envs
from stable_baselines3 import SAC
from envs.shielding_wrapper import ShieldingWrapper

def run_episode(env, model, max_steps=300):
    obs, _ = env.reset()
    done, total_reward = False, 0
    step_count = 0

    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1

    print(f"→ Blocages de sécurité : {env.shield_count}")
    for entry in env.shield_log:
        print(f"  - θ = {entry['angle_deg']:.1f}°, action = {entry['original_action']:.2f}")
    return total_reward

def main():
    model = SAC.load("models/sac_pendulum")
    base_env = gym.make("PendulumDangerous-v1", render_mode=None)
    env = ShieldingWrapper(base_env)

    rewards = []
    for i in range(5):
        print(f"\n--- Épisode {i+1} ---")
        r = run_episode(env, model)
        rewards.append(r)

    print(f"\nRécompense moyenne sur 5 épisodes : {np.mean(rewards):.2f}")

if __name__ == "__main__":
    import numpy as np
    main()
