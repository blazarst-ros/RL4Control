import time

import gymnasium as gym


def main() -> None:
    env = gym.make("CartPole-v1", render_mode="human")
    observation, info = env.reset(seed=42)

    for _ in range(300):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        time.sleep(0.02)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    print("CartPole rendering test: PASSED")


if __name__ == "__main__":
    main()
