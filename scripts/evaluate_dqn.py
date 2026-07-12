from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "cartpole_dqn"


def main() -> None:
    if not MODEL_PATH.with_suffix(".zip").exists():
        raise FileNotFoundError(
            f"没有找到模型：{MODEL_PATH.with_suffix('.zip')}"
        )

    env = gym.make("CartPole-v1", render_mode="human")
    model = DQN.load(str(MODEL_PATH))

    observation, info = env.reset(seed=42)
    episode_reward = 0.0

    while True:
        action, _ = model.predict(observation, deterministic=True)

        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += float(reward)

        if terminated or truncated:
            break

    env.close()
    print("Episode reward:", episode_reward)


if __name__ == "__main__":
    main()
