import argparse
import os
from collections import defaultdict

import pandas as pd
import wandb


def wandb_summary(args):
    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)
    run = wandb.init(
        entity=args.wandb_org,
        project=args.wandb_project,
        name=args.wandb_id if args.wandb_id else args.wandb_run_name,
        resume="auto",
        id=args.wandb_id,
    )
    for path in args.collect_paths:
        if "math" in path:
            task = "math"
        elif "simple" in path:
            task = "general"
        elif "game" in path:
            task = "game"

        avg = defaultdict(list)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        df = pd.read_csv(f"{path}/metrics.csv")
        for _, row in df.iterrows():
            key = row.get("key")
            if "eval-" not in key:
                key = f"eval-{key}"
            value = row.get("value")
            if pd.isna(key) or pd.isna(value):
                continue
            wandb.log({key: value}, step=args.step)
            avg[key.split("/")[-1]].append(value)
        for key, values in avg.items():
            wandb.log({f"eval-{task}/{key}": sum(values) / len(values)}, step=args.step)
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log summary results from CSVs to wandb.")
    parser.add_argument("--collect_paths", type=str, required=True, nargs="+", help="CSV files to log.")
    parser.add_argument("--wandb_project", type=str, default="oat-game-eval")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default="stlm")
    parser.add_argument("--wandb_api_key", type=str, default=None)
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()
    wandb_summary(args)
