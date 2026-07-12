from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs" / "dqn"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    env = Monitor(gym.make("CartPole-v1"))

    checkpoint_callback = CheckpointCallback(
        save_freq=5_000,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="cartpole_dqn",
        save_replay_buffer=True,
    )

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.2,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
        device="auto",
        seed=42,
    )

    model.learn(
        total_timesteps=50_000,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    model_path = MODEL_DIR / "cartpole_dqn"
    model.save(str(model_path))

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=10,
        deterministic=True,
    )

    env.close()

    print(f"Model saved to: {model_path}.zip")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
