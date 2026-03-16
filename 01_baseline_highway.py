import argparse
import json
import os
import random

import numpy as np
import pandas as pd

import config
from vehicle_env import VehicleEnv


SCENARIO = "Highway"


def evaluate_threshold(action_index, episodes, seed):
    np.random.seed(seed)
    random.seed(seed)

    env = VehicleEnv(scenario=SCENARIO)

    rewards = []
    ep_hfr = []
    ep_ppr = []
    ep_ehr = []

    total_attempted = 0
    total_success = 0
    total_failed = 0
    total_pingpong = 0
    total_weak_signal = 0

    for _ in range(episodes):
        env.reset()
        done = False
        total_reward = 0.0

        while not done:
            _, reward, done, _ = env.step(action_index)
            total_reward += float(reward)

        stats = env.get_episode_stats()
        rewards.append(total_reward)
        ep_hfr.append(stats["hfr"])
        ep_ppr.append(stats["ppr"])
        ep_ehr.append(stats["ehr"])

        total_attempted += int(stats["ho_attempted"])
        total_success += int(stats["ho_success"])
        total_failed += int(stats["ho_failed"])
        total_pingpong += int(stats["pingpong"])
        total_weak_signal += int(stats["weak_signal_event"])

    agg_hfr = total_failed / max(total_attempted, 1)
    agg_ppr = total_pingpong / max(total_attempted, 1)
    agg_ehr = 1.0 - agg_hfr - agg_ppr

    return {
        "scenario": SCENARIO,
        "threshold_dbm": config.ACTION_THRESHOLDS[action_index],
        "episodes": int(episodes),
        "mean_reward": float(np.mean(rewards)),
        "mean_episode_hfr": float(np.mean(ep_hfr)),
        "mean_episode_ppr": float(np.mean(ep_ppr)),
        "mean_episode_ehr": float(np.mean(ep_ehr)),
        "aggregate_hfr": float(agg_hfr),
        "aggregate_ppr": float(agg_ppr),
        "aggregate_ehr": float(agg_ehr),
        "total_ho_attempted": int(total_attempted),
        "total_ho_success": int(total_success),
        "total_ho_failed": int(total_failed),
        "total_pingpong": int(total_pingpong),
        "total_weak_signal_event": int(total_weak_signal),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=config.BASELINE_EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=config.BASELINE_SEED)
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = os.path.join("Result", "highway", "baseline")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for action_index, threshold in enumerate(config.ACTION_THRESHOLDS):
        row = evaluate_threshold(
            action_index=action_index,
            episodes=args.episodes,
            seed=args.seed,
        )
        rows.append(row)
        print(
            f"[Baseline][{SCENARIO}] threshold={threshold} "
            f"reward={row['mean_reward']:.3f} "
            f"HFR={row['aggregate_hfr']:.4f} "
            f"PPR={row['aggregate_ppr']:.4f} "
            f"EHR={row['aggregate_ehr']:.4f}"
        )

    metrics_df = pd.DataFrame(rows).sort_values("threshold_dbm").reset_index(drop=True)
    metrics_path = os.path.join(out_dir, "threshold_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    best_idx = int(metrics_df["mean_reward"].idxmax())
    best_row = metrics_df.iloc[best_idx].to_dict()
    best_row["best_rule"] = "max_mean_reward"

    best_path = os.path.join(out_dir, "best_baseline.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_row, f, indent=2)

    print(f"Saved: {metrics_path}")
    print(f"Saved: {best_path}")
    print(
        f"Best threshold={best_row['threshold_dbm']} dBm, "
        f"reward={best_row['mean_reward']:.3f}, "
        f"HFR={best_row['aggregate_hfr']:.4f}, "
        f"PPR={best_row['aggregate_ppr']:.4f}, "
        f"EHR={best_row['aggregate_ehr']:.4f}"
    )


if __name__ == "__main__":
    main()
