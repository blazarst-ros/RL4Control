from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(
    "/media/blazarst/86EAF6D5EAF6C08B/Course Resources/自动控制原理/"
    "RL for Inverted Pendulum System/Data"
)
DEFAULT_SWEEP_NAME = "cartpole_dqn_report"

# Baseline values copied from scripts/train_dqn.py.
# These stay fixed unless the user explicitly changes them in the CLI.
BASE_LEARNING_RATE = 1e-3
BASE_GAMMA = 0.99
BASE_EXPLORATION_FRACTION = 0.2
BASE_EXPLORATION_INITIAL_EPS = 1.0
BASE_EXPLORATION_FINAL_EPS = 0.05
BASE_BUFFER_SIZE = 50_000
BASE_LEARNING_STARTS = 1_000
BASE_BATCH_SIZE = 64
BASE_TRAIN_FREQ = 4
BASE_TARGET_UPDATE_INTERVAL = 1_000
BASE_TOTAL_TIMESTEPS = 50_000
BASE_EVAL_FREQ = 5_000
BASE_EVAL_EPISODES = 10
BASE_CHECKPOINT_FREQ = 5_000
BASE_SEED = 42


@dataclass(frozen=True)
class RunConfig:
    """Configuration for one training run."""

    group: str
    setting_name: str
    learning_rate: float = BASE_LEARNING_RATE
    gamma: float = BASE_GAMMA
    exploration_fraction: float = BASE_EXPLORATION_FRACTION
    exploration_initial_eps: float = BASE_EXPLORATION_INITIAL_EPS
    exploration_final_eps: float = BASE_EXPLORATION_FINAL_EPS
    exploration_mode: str = "epsilon_greedy"
    seed: int = BASE_SEED
    total_timesteps: int = BASE_TOTAL_TIMESTEPS
    eval_freq: int = BASE_EVAL_FREQ
    eval_episodes: int = BASE_EVAL_EPISODES
    checkpoint_freq: int = BASE_CHECKPOINT_FREQ
    buffer_size: int = BASE_BUFFER_SIZE
    learning_starts: int = BASE_LEARNING_STARTS
    batch_size: int = BASE_BATCH_SIZE
    train_freq: int = BASE_TRAIN_FREQ
    target_update_interval: int = BASE_TARGET_UPDATE_INTERVAL

    def to_dqn_kwargs(self) -> dict[str, Any]:
        """Convert this config into the keyword args expected by SB3 DQN."""
        if self.exploration_mode == "greedy":
            exploration_fraction = 0.0
            exploration_initial_eps = 0.0
            exploration_final_eps = 0.0
        else:
            exploration_fraction = self.exploration_fraction
            exploration_initial_eps = self.exploration_initial_eps
            exploration_final_eps = self.exploration_final_eps

        return {
            "policy": "MlpPolicy",
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "train_freq": self.train_freq,
            "target_update_interval": self.target_update_interval,
            "exploration_fraction": exploration_fraction,
            "exploration_initial_eps": exploration_initial_eps,
            "exploration_final_eps": exploration_final_eps,
            "verbose": 1,
            "tensorboard_log": None,
            "device": "auto",
            "seed": self.seed,
        }


@dataclass
class RunPaths:
    """Folder layout for one run."""

    run_dir: Path
    metrics_dir: Path
    plots_dir: Path
    checkpoints_dir: Path
    eval_dir: Path
    tensorboard_dir: Path
    models_dir: Path

    @classmethod
    def create(cls, run_dir: Path) -> "RunPaths":
        """Create the directory tree for one run and return the paths."""
        metrics_dir = run_dir / "metrics"
        plots_dir = run_dir / "plots"
        checkpoints_dir = run_dir / "checkpoints"
        eval_dir = run_dir / "evaluation"
        tensorboard_dir = run_dir / "tensorboard"
        models_dir = run_dir / "models"
        for directory in (
            run_dir,
            metrics_dir,
            plots_dir,
            checkpoints_dir,
            eval_dir,
            tensorboard_dir,
            models_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return cls(
            run_dir=run_dir,
            metrics_dir=metrics_dir,
            plots_dir=plots_dir,
            checkpoints_dir=checkpoints_dir,
            eval_dir=eval_dir,
            tensorboard_dir=tensorboard_dir,
            models_dir=models_dir,
        )


def timestamp_label() -> str:
    """Create a compact timestamp used for archive folder names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_float_label(value: float) -> str:
    """Make a float safe for folder names."""
    return f"{value:g}".replace(".", "p").replace("-", "m")


def ensure_data_root(path: Path) -> Path:
    """Create the archive root and guard the default external mount."""
    path = path.expanduser()
    if path == DEFAULT_DATA_ROOT:
        mount_root = Path("/media/blazarst/86EAF6D5EAF6C08B")
        if not mount_root.exists():
            raise FileNotFoundError(
                "The default archive mount is unavailable: "
                f"{mount_root}. Mount the external drive or pass --data-root."
            )
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    """Write a pretty JSON file with UTF-8 encoding."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_csv_safe(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a CSV file, returning an empty frame if the file is missing."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def moving_average(values: Iterable[float], window: int = 10) -> np.ndarray:
    """Compute a simple moving average for smoothing plots."""
    series = pd.Series(list(values), dtype=float)
    if series.empty:
        return np.array([], dtype=float)
    return series.rolling(window=window, min_periods=1).mean().to_numpy()


def make_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    """Build a plain Markdown table without extra dependencies."""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return [header, separator, *body] if rows else [header, separator]


class MetricRecorderCallback(BaseCallback):
    """Collect training-step and episode-level metrics during learning."""

    def __init__(self, sample_every: int = 1) -> None:
        super().__init__()
        self.sample_every = sample_every
        self.step_rows: list[dict[str, Any]] = []
        self.episode_rows: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        # Record step-level metrics that help explain how the agent learns.
        if self.n_calls % self.sample_every == 0:
            logger_values = getattr(self.model.logger, "name_to_value", {})
            exploration_rate = getattr(self.model, "exploration_rate", float("nan"))
            loss = logger_values.get("train/loss")
            learning_rate = logger_values.get("train/learning_rate")
            self.step_rows.append(
                {
                    "step": int(self.num_timesteps),
                    "exploration_rate": float(exploration_rate)
                    if exploration_rate is not None
                    else float("nan"),
                    "loss": float(loss) if loss is not None else float("nan"),
                    "learning_rate": float(learning_rate)
                    if learning_rate is not None
                    else float("nan"),
                }
            )

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if not isinstance(infos, (list, tuple)):
            infos = [infos]
        if not isinstance(dones, (list, tuple, np.ndarray)):
            dones = [dones]

        for info, done in zip(infos, dones):
            if done and isinstance(info, dict) and "episode" in info:
                episode_info = info["episode"]
                self.episode_rows.append(
                    {
                        "step": int(self.num_timesteps),
                        "episode_reward": float(episode_info["r"]),
                        "episode_length": int(episode_info["l"]),
                    }
                )

        return True


def plot_line(
    x: Iterable[float],
    y: Iterable[float],
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    moving_average_window: int | None = None,
) -> None:
    """Save a line chart with optional smoothing."""
    x_values = np.asarray(list(x), dtype=float)
    y_values = np.asarray(list(y), dtype=float)
    if x_values.size == 0 or y_values.size == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_values, y_values, linewidth=1.4, alpha=0.8, label="raw")
    if moving_average_window and y_values.size >= 2:
        ax.plot(
            x_values,
            moving_average(y_values, window=moving_average_window),
            linewidth=2.0,
            label=f"moving average ({moving_average_window})",
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_evaluation_curve(eval_frame: pd.DataFrame, path: Path, title: str) -> None:
    """Plot evaluation reward with a standard-deviation band."""
    if eval_frame.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(eval_frame["timesteps"], eval_frame["mean_reward"], marker="o", linewidth=1.8)
    ax.fill_between(
        eval_frame["timesteps"],
        eval_frame["mean_reward"] - eval_frame["std_reward"],
        eval_frame["mean_reward"] + eval_frame["std_reward"],
        alpha=0.2,
    )
    ax.set_title(title)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Evaluation reward")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def load_eval_history(eval_dir: Path) -> pd.DataFrame:
    """Load EvalCallback output and reshape it into a summary table."""
    npz_path = eval_dir / "evaluations.npz"
    if not npz_path.exists():
        return pd.DataFrame(columns=["timesteps", "mean_reward", "std_reward", "min_reward", "max_reward"])

    data = np.load(npz_path)
    timesteps = np.asarray(data["timesteps"], dtype=float).reshape(-1)
    results = np.asarray(data["results"], dtype=float)
    if results.size == 0:
        return pd.DataFrame(columns=["timesteps", "mean_reward", "std_reward", "min_reward", "max_reward"])
    if results.ndim == 1:
        results = results.reshape(-1, 1)

    return pd.DataFrame(
        {
            "timesteps": timesteps,
            "mean_reward": results.mean(axis=1),
            "std_reward": results.std(axis=1),
            "min_reward": results.min(axis=1),
            "max_reward": results.max(axis=1),
        }
    )


def save_monitor_csv(run_dir: Path, episode_rows: list[dict[str, Any]]) -> Path:
    """Write episode-level training history to disk."""
    path = run_dir / "metrics" / "training_episodes.csv"
    frame = pd.DataFrame(
        episode_rows,
        columns=["step", "episode_reward", "episode_length"],
    )
    if not frame.empty:
        frame = frame.sort_values("step").reset_index(drop=True)
    frame.to_csv(path, index=False)
    return path


def save_step_csv(run_dir: Path, step_rows: list[dict[str, Any]]) -> Path:
    """Write step-level training history to disk."""
    path = run_dir / "metrics" / "training_steps.csv"
    frame = pd.DataFrame(
        step_rows,
        columns=["step", "exploration_rate", "loss", "learning_rate"],
    )
    if not frame.empty:
        frame = frame.sort_values("step").reset_index(drop=True)
    frame.to_csv(path, index=False)
    return path


def create_run_plots(run_paths: RunPaths) -> None:
    """Generate the plots for one run."""
    episode_frame = read_csv_safe(
        run_paths.metrics_dir / "training_episodes.csv",
        ["step", "episode_reward", "episode_length"],
    )
    step_frame = read_csv_safe(
        run_paths.metrics_dir / "training_steps.csv",
        ["step", "exploration_rate", "loss", "learning_rate"],
    )
    eval_frame = load_eval_history(run_paths.eval_dir)

    if not episode_frame.empty:
        plot_line(
            episode_frame["step"],
            episode_frame["episode_reward"],
            run_paths.plots_dir / "training_reward_curve.png",
            "Training Reward vs Steps",
            "Timesteps",
            "Episode reward",
        )
        plot_line(
            range(len(episode_frame)),
            episode_frame["episode_reward"],
            run_paths.plots_dir / "reward_by_episode.png",
            "Training Reward vs Episode",
            "Episode index",
            "Episode reward",
            moving_average_window=10,
        )
        plot_line(
            episode_frame["step"],
            episode_frame["episode_length"],
            run_paths.plots_dir / "episode_length_curve.png",
            "Episode Length vs Steps",
            "Timesteps",
            "Episode length",
        )

    if not step_frame.empty:
        plot_line(
            step_frame["step"],
            step_frame["loss"],
            run_paths.plots_dir / "loss_curve.png",
            "Training Loss vs Steps",
            "Timesteps",
            "Loss",
        )
        plot_line(
            step_frame["step"],
            step_frame["exploration_rate"],
            run_paths.plots_dir / "exploration_rate_curve.png",
            "Exploration Rate vs Steps",
            "Timesteps",
            "Exploration rate",
        )

    plot_evaluation_curve(
        eval_frame,
        run_paths.plots_dir / "evaluation_reward_curve.png",
        "Evaluation Reward vs Training Timesteps",
    )

    if not eval_frame.empty:
        eval_frame.to_csv(run_paths.metrics_dir / "evaluation_history.csv", index=False)


def experiment_name(config: RunConfig) -> str:
    """Create a readable folder name for a run."""
    return (
        f"{config.setting_name}"
        f"_seed_{config.seed}"
    )


def run_single_experiment(config: RunConfig, run_paths: RunPaths) -> dict[str, Any]:
    """Train one DQN model, evaluate it, and save all artifacts."""
    training_env = Monitor(
        gym.make("CartPole-v1"),
        filename=str(run_paths.run_dir / "monitor.csv"),
    )
    eval_env = Monitor(
        gym.make("CartPole-v1"),
        filename=str(run_paths.eval_dir / "eval_monitor.csv"),
    )

    dqn_kwargs = config.to_dqn_kwargs()
    dqn_kwargs["tensorboard_log"] = str(run_paths.tensorboard_dir)

    model = DQN(env=training_env, **dqn_kwargs)

    recorder = MetricRecorderCallback(sample_every=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=str(run_paths.checkpoints_dir),
        name_prefix="cartpole_dqn",
        save_replay_buffer=True,
    )
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(run_paths.models_dir),
        log_path=str(run_paths.eval_dir),
        n_eval_episodes=config.eval_episodes,
        eval_freq=config.eval_freq,
        deterministic=True,
        render=False,
        verbose=1,
    )

    callback = CallbackList([recorder, checkpoint_callback, eval_callback])
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    model.save(str(run_paths.models_dir / "final_model"))

    eval_rewards, eval_lengths = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=config.eval_episodes,
        deterministic=True,
        return_episode_rewards=True,
    )

    training_env.close()
    eval_env.close()

    save_step_csv(run_paths.run_dir, recorder.step_rows)
    save_monitor_csv(run_paths.run_dir, recorder.episode_rows)
    create_run_plots(run_paths)

    final_eval_mean = float(np.mean(eval_rewards)) if eval_rewards else float("nan")
    final_eval_std = float(np.std(eval_rewards)) if eval_rewards else float("nan")
    summary = {
        "config": asdict(config),
        "run_name": run_paths.run_dir.name,
        "final_eval_mean_reward": final_eval_mean,
        "final_eval_std_reward": final_eval_std,
        "final_eval_episode_rewards": [float(value) for value in eval_rewards],
        "final_eval_episode_lengths": [int(value) for value in eval_lengths],
        "artifact_paths": {
            "run_dir": str(run_paths.run_dir),
            "plots_dir": str(run_paths.plots_dir),
            "models_dir": str(run_paths.models_dir),
            "checkpoints_dir": str(run_paths.checkpoints_dir),
            "evaluation_dir": str(run_paths.eval_dir),
        },
    }
    write_json(run_paths.run_dir / "summary.json", summary)
    return summary


def aggregate_comparison(group_dir: Path, run_summaries: list[dict[str, Any]]) -> None:
    """Combine all runs in one comparison group into summary files and plots."""
    comparison_dir = group_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    eval_fig, eval_ax = plt.subplots(figsize=(10, 5))
    reward_fig, reward_ax = plt.subplots(figsize=(10, 5))
    loss_fig, loss_ax = plt.subplots(figsize=(10, 5))
    exploration_fig, exploration_ax = plt.subplots(figsize=(10, 5))

    for summary in run_summaries:
        run_dir = Path(summary["artifact_paths"]["run_dir"])
        config = summary["config"]
        label = summary["run_name"]

        episode_frame = read_csv_safe(
            run_dir / "metrics" / "training_episodes.csv",
            ["step", "episode_reward", "episode_length"],
        )
        step_frame = read_csv_safe(
            run_dir / "metrics" / "training_steps.csv",
            ["step", "exploration_rate", "loss", "learning_rate"],
        )
        eval_frame = load_eval_history(run_dir / "evaluation")

        summary_rows.append(
            {
                "run_name": summary["run_name"],
                "learning_rate": config["learning_rate"],
                "gamma": config["gamma"],
                "exploration_mode": config["exploration_mode"],
                "exploration_fraction": config["exploration_fraction"],
                "seed": config["seed"],
                "final_eval_mean_reward": summary["final_eval_mean_reward"],
                "final_eval_std_reward": summary["final_eval_std_reward"],
                "final_training_reward": float(episode_frame["episode_reward"].iloc[-1])
                if not episode_frame.empty
                else float("nan"),
                "final_training_loss": float(step_frame["loss"].dropna().iloc[-1])
                if not step_frame.empty and not step_frame["loss"].dropna().empty
                else float("nan"),
                "best_eval_mean_reward": float(eval_frame["mean_reward"].max())
                if not eval_frame.empty
                else float("nan"),
            }
        )

        if not eval_frame.empty:
            eval_ax.plot(
                eval_frame["timesteps"],
                eval_frame["mean_reward"],
                marker="o",
                linewidth=1.6,
                label=label,
            )
        if not episode_frame.empty:
            reward_ax.plot(
                episode_frame["step"],
                moving_average(episode_frame["episode_reward"], window=10),
                linewidth=1.8,
                label=label,
            )
        if not step_frame.empty:
            loss_values = step_frame["loss"].dropna()
            loss_steps = step_frame.loc[loss_values.index, "step"]
            if not loss_values.empty:
                loss_ax.plot(loss_steps, loss_values, linewidth=1.4, label=label)
            exploration_ax.plot(
                step_frame["step"],
                step_frame["exploration_rate"],
                linewidth=1.4,
                label=label,
            )

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame = summary_frame.sort_values(
        by=["final_eval_mean_reward", "best_eval_mean_reward"],
        ascending=False,
    )
    summary_frame.to_csv(comparison_dir / "summary.csv", index=False)
    write_json(
        comparison_dir / "summary.json",
        {
            "runs": summary_rows,
            "best_run": summary_frame.iloc[0].to_dict() if not summary_frame.empty else {},
        },
    )

    best_row = summary_frame.iloc[0] if not summary_frame.empty else None
    md_lines = [
        "# CartPole DQN Comparison",
        "",
    ]
    if best_row is not None:
        md_lines.extend(
            [
                f"- Best run: `{best_row['run_name']}`",
                f"- Final evaluation reward: `{best_row['final_eval_mean_reward']:.3f}`",
                f"- Best evaluation mean reward: `{best_row['best_eval_mean_reward']:.3f}`",
                "",
            ]
        )
    md_lines.extend(
        [
            "## Top Configurations",
            "",
            *make_markdown_table(
                summary_frame.head(5).to_dict(orient="records"),
                [
                    "run_name",
                    "learning_rate",
                    "gamma",
                    "exploration_mode",
                    "exploration_fraction",
                    "final_eval_mean_reward",
                    "best_eval_mean_reward",
                ],
            ),
            "",
        ]
    )
    (comparison_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    for ax, title, xlabel, ylabel, filename in (
        (eval_ax, "Evaluation Reward Comparison", "Timesteps", "Mean reward", "evaluation_reward_comparison.png"),
        (reward_ax, "Training Reward Comparison", "Timesteps", "Moving average reward", "training_reward_comparison.png"),
        (loss_ax, "Training Loss Comparison", "Timesteps", "Loss", "loss_comparison.png"),
        (
            exploration_ax,
            "Exploration Rate Comparison",
            "Timesteps",
            "Exploration rate",
            "exploration_rate_comparison.png",
        ),
    ):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if ax.lines:
            ax.legend(fontsize=8)
        ax.figure.tight_layout()
        ax.figure.savefig(comparison_dir / filename, dpi=160)
        plt.close(ax.figure)


def build_argument_parser() -> argparse.ArgumentParser:
    """Define the CLI for the experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run CartPole DQN comparison experiments and archive the results."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory that receives the archived outputs.",
    )
    parser.add_argument(
        "--sweep-name",
        default=DEFAULT_SWEEP_NAME,
        help="Top-level folder name inside the data root.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["learning_rate", "gamma", "exploration"],
        default=["learning_rate", "gamma", "exploration"],
        help="Which comparison groups to run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[BASE_SEED],
        help="Random seeds to use.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=BASE_TOTAL_TIMESTEPS,
        help="Training steps per run.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=BASE_EVAL_FREQ,
        help="Evaluation interval in timesteps.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=BASE_EVAL_EPISODES,
        help="Evaluation episodes per checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=BASE_CHECKPOINT_FREQ,
        help="Model checkpoint interval in timesteps.",
    )
    return parser


def learning_rate_specs(args: argparse.Namespace) -> list[RunConfig]:
    """Create the learning-rate comparison group."""
    values = [5e-4, BASE_LEARNING_RATE, 2e-3, 5e-3]
    return [
        RunConfig(
            group="learning_rate",
            setting_name=f"lr_{safe_float_label(value)}",
            learning_rate=value,
            seed=seed,
            total_timesteps=args.total_timesteps,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            checkpoint_freq=args.checkpoint_freq,
        )
        for value in values
        for seed in args.seeds
    ]


def gamma_specs(args: argparse.Namespace) -> list[RunConfig]:
    """Create the gamma comparison group."""
    values = [0.90, 0.95, BASE_GAMMA, 0.995]
    return [
        RunConfig(
            group="gamma",
            setting_name=f"gamma_{safe_float_label(value)}",
            gamma=value,
            seed=seed,
            total_timesteps=args.total_timesteps,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            checkpoint_freq=args.checkpoint_freq,
        )
        for value in values
        for seed in args.seeds
    ]


def exploration_specs(args: argparse.Namespace) -> list[RunConfig]:
    """Create the exploration comparison group."""
    settings = [
        ("baseline_eps_0p2", "epsilon_greedy", 0.2),
        ("fast_decay", "epsilon_greedy", 0.1),
        ("slow_decay", "epsilon_greedy", 0.4),
        ("greedy", "greedy", 0.0),
    ]
    return [
        RunConfig(
            group="exploration",
            setting_name=name,
            exploration_mode=mode,
            exploration_fraction=fraction,
            seed=seed,
            total_timesteps=args.total_timesteps,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            checkpoint_freq=args.checkpoint_freq,
        )
        for name, mode, fraction in settings
        for seed in args.seeds
    ]


def build_group_specs(args: argparse.Namespace) -> dict[str, list[RunConfig]]:
    """Build the selected comparison groups."""
    specs: dict[str, list[RunConfig]] = {}
    if "learning_rate" in args.groups:
        specs["learning_rate"] = learning_rate_specs(args)
    if "gamma" in args.groups:
        specs["gamma"] = gamma_specs(args)
    if "exploration" in args.groups:
        specs["exploration"] = exploration_specs(args)
    return specs


def main() -> None:
    """Run all selected experiments and write the archive tree."""
    args = build_argument_parser().parse_args()
    data_root = ensure_data_root(args.data_root)
    sweep_root = data_root / args.sweep_name / timestamp_label()
    sweep_root.mkdir(parents=True, exist_ok=True)

    write_json(
        sweep_root / "sweep_config.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "project_root": str(PROJECT_ROOT),
            "data_root": str(data_root),
            "groups": args.groups,
            "seeds": args.seeds,
            "defaults": {
                "learning_rate": BASE_LEARNING_RATE,
                "gamma": BASE_GAMMA,
                "exploration_fraction": BASE_EXPLORATION_FRACTION,
                "exploration_initial_eps": BASE_EXPLORATION_INITIAL_EPS,
                "exploration_final_eps": BASE_EXPLORATION_FINAL_EPS,
                "buffer_size": BASE_BUFFER_SIZE,
                "learning_starts": BASE_LEARNING_STARTS,
                "batch_size": BASE_BATCH_SIZE,
                "train_freq": BASE_TRAIN_FREQ,
                "target_update_interval": BASE_TARGET_UPDATE_INTERVAL,
                "total_timesteps": args.total_timesteps,
                "eval_freq": args.eval_freq,
                "eval_episodes": args.eval_episodes,
                "checkpoint_freq": args.checkpoint_freq,
            },
        },
    )

    for group_name, configs in build_group_specs(args).items():
        group_dir = sweep_root / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        run_summaries: list[dict[str, Any]] = []

        for config in configs:
            run_dir = group_dir / "runs" / experiment_name(config)
            run_paths = RunPaths.create(run_dir)
            write_json(run_paths.run_dir / "config.json", asdict(config))
            run_summaries.append(run_single_experiment(config, run_paths))

        aggregate_comparison(group_dir, run_summaries)

    print(f"Experiment archive written to: {sweep_root}")


if __name__ == "__main__":
    main()
