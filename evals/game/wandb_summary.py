import os

import pandas as pd
import wandb
from tabulate import tabulate


def load_csv(filepath):
    """Loads a CSV file and returns a pandas DataFrame or None if not found."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        print(f"[Warning] File not found: {filepath}")
        return None

def calculate_evaluation_stats(df_eval):
    """Calculates evaluation metrics from the CSV DataFrame."""
    if df_eval is None or df_eval.empty:
        return {}

    def eval_metrics_for(df, prefix):
        outcomes = df["model_outcome"].value_counts(normalize=True).to_dict()
        return {
            f"{prefix}/num_episodes": len(df),
            f"{prefix}/avg_eps_duration": df["t_delta"].mean(),
            f"{prefix}/avg_num_turns": df["num_turns"].mean(),
            f"{prefix}/avg_model_reward": df["model_reward"].mean(),
            f"{prefix}/avg_opponent_reward": df["opponent_reward"].mean(),
            f"{prefix}/win_rate": outcomes.get("win", 0),
            f"{prefix}/draw_rate": outcomes.get("draw", 0),
            f"{prefix}/loss_rate": outcomes.get("loss", 0),
        }

    metrics = {}
    # metrics.update(eval_metrics_for(df_eval, "eval/overall"))

    for env_id in df_eval["env_id"].unique():
        for opponent_name in df_eval["opponent_name"].unique():
            env_df = df_eval[
                (df_eval["env_id"] == env_id) & (df_eval["opponent_name"] == opponent_name)
            ]
            metrics.update(eval_metrics_for(env_df, f"eval-{env_id}-{opponent_name}"))

    return metrics


def pretty_print_table(metrics_dict, title):
    if not metrics_dict:
        print(f"\n{title} - No Data Available")
        return

    table_data = [
        (key, f"{value:.4f}" if isinstance(value, float) else value)
        for key, value in metrics_dict.items()
    ]
    table = tabulate(table_data, headers=["Metric", "Value"], tablefmt="fancy_grid")
    print(f"\n{title}\n{table}")


def analyze(
        data_folder: str,
        wandb_org: str = "stlm",
        wandb_project: str = "oat-game-eval",
        wandb_run_name: str = None,
        wandb_id: str = None,
        step: int = None,
        no_wandb: bool = False,
):
    if not no_wandb:
        wandb.init(
            entity=wandb_org,
            project=wandb_project,
            name=wandb_run_name,
            resume="auto",
            id=wandb_id,
        )

    df_eval = load_csv(
        os.path.join(data_folder, "logging", f"eval_info.csv")
    )

    eval_metrics = calculate_evaluation_stats(df_eval)

    pretty_print_table(eval_metrics, f"Evaluation Metrics")

    if not no_wandb:
        wandb.log(eval_metrics, step=step)
        wandb.finish()
