import gymnasium as gym


def main() -> None:
    env = gym.make("CartPole-v1")

    observation, info = env.reset(seed=42)
    total_reward = 0.0

    print("Observation shape:", observation.shape)
    print("Observation:", observation)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    for step in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        if terminated or truncated:
            print(f"Episode finished at step {step + 1}")
            break

    env.close()
    print("Total reward:", total_reward)
    print("CartPole environment test: PASSED")


if __name__ == "__main__":
    main()
