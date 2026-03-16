import argparse
import json
import os
import random

import numpy as np
import pandas as pd

import config
from vehicle_env import VehicleEnv


SCENARIO = "Highway"


def safe_nanmean(values):
    s = pd.Series(values, dtype=float)
    if s.notna().any():
        return float(s.mean(skipna=True))
    return float("nan")


def estimate_serving_rsrp_range(episodes, seed, action_index=3):
    np.random.seed(seed)
    random.seed(seed)

    env = VehicleEnv(scenario=SCENARIO)
    vals = []
    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            _, _, done, info = env.step(action_index)
            vals.append(float(info["rsrp"]))

    arr = np.asarray(vals, dtype=float)
    return {
        "rsrp_min": float(np.min(arr)),
        "rsrp_max": float(np.max(arr)),
        "rsrp_mean": float(np.mean(arr)),
    }


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
    no_attempt_episode_count = 0
    total_pending_validation_started = 0
    total_validation_success = 0
    total_validation_failure = 0

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
        no_attempt_episode_count += int(stats["no_attempt_episode"])
        total_pending_validation_started += int(stats["pending_validation_started"])
        total_validation_success += int(stats["validation_success"])
        total_validation_failure += int(stats["validation_failure"])

    agg_hfr = total_failed / max(total_attempted, 1)
    agg_ppr = total_pingpong / max(total_attempted, 1)
    agg_ehr = 1.0 - agg_hfr - agg_ppr

    return {
        "scenario": SCENARIO,
        "threshold_dbm": config.ACTION_THRESHOLDS[action_index],
        "episodes": int(episodes),
        "mean_reward": float(np.mean(rewards)),
        "mean_episode_hfr": safe_nanmean(ep_hfr),
        "mean_episode_ppr": safe_nanmean(ep_ppr),
        "mean_episode_ehr": safe_nanmean(ep_ehr),
        "aggregate_hfr": float(agg_hfr),
        "aggregate_ppr": float(agg_ppr),
        "aggregate_ehr": float(agg_ehr),
        "total_ho_attempted": int(total_attempted),
        "total_ho_success": int(total_success),
        "total_ho_failed": int(total_failed),
        "total_pingpong": int(total_pingpong),
        "total_weak_signal_event": int(total_weak_signal),
        "no_attempt_episode_count": int(no_attempt_episode_count),
        "no_attempt_episode_ratio": float(no_attempt_episode_count / max(episodes, 1)),
        "total_pending_validation_started": int(total_pending_validation_started),
        "total_validation_success": int(total_validation_success),
        "total_validation_failure": int(total_validation_failure),
    }


def select_best_baseline(metrics_df):
    valid_df = metrics_df[
        metrics_df["total_ho_attempted"] >= int(config.BASELINE_MIN_TOTAL_ATTEMPTS)
    ].copy()

    if valid_df.empty:
        ranked = metrics_df.sort_values(
            by=["mean_reward"],
            ascending=[False],
        )
        best_row = ranked.iloc[0].to_dict()
        selection_mode = "fallback_reward"
    else:
        ranked = valid_df.sort_values(
            by=["aggregate_ehr", "aggregate_hfr", "aggregate_ppr", "mean_reward"],
            ascending=[False, True, True, False],
        )
        best_row = ranked.iloc[0].to_dict()
        selection_mode = "filtered_kpi_priority"

    best_row["selection_mode"] = selection_mode
    best_row["baseline_min_total_attempts"] = int(config.BASELINE_MIN_TOTAL_ATTEMPTS)
    return best_row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=config.BASELINE_EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=config.BASELINE_SEED)
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = os.path.join("Result", "highway", "baseline")
    os.makedirs(out_dir, exist_ok=True)

    rsrp_stats = estimate_serving_rsrp_range(
        episodes=max(10, args.episodes // 5),
        seed=args.seed,
    )
    print(
        f"[Baseline][{SCENARIO}] serving RSRP range "
        f"min={rsrp_stats['rsrp_min']:.2f}, max={rsrp_stats['rsrp_max']:.2f}, "
        f"mean={rsrp_stats['rsrp_mean']:.2f}, weak_threshold={config.WEAK_SIGNAL_THRESHOLD_DBM:.1f}"
    )

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
            f"agg_HFR={row['aggregate_hfr']:.4f} "
            f"agg_PPR={row['aggregate_ppr']:.4f} "
            f"agg_EHR={row['aggregate_ehr']:.4f} "
            f"attempts={row['total_ho_attempted']} "
            f"weak={row['total_weak_signal_event']} "
            f"val_start={row['total_pending_validation_started']} "
            f"val_succ={row['total_validation_success']} "
            f"val_fail={row['total_validation_failure']} "
            f"no_attempt_ratio={row['no_attempt_episode_ratio']:.3f}"
        )

    metrics_df = pd.DataFrame(rows).sort_values("threshold_dbm").reset_index(drop=True)
    metrics_path = os.path.join(out_dir, "threshold_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    best_row = select_best_baseline(metrics_df)
    best_path = os.path.join(out_dir, "best_baseline.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_row, f, indent=2)

    print(f"Saved: {metrics_path}")
    print(f"Saved: {best_path}")
    print(
        f"Best threshold={best_row['threshold_dbm']} dBm, "
        f"selection_mode={best_row['selection_mode']}, "
        f"reward={best_row['mean_reward']:.3f}, "
        f"agg_HFR={best_row['aggregate_hfr']:.4f}, "
        f"agg_PPR={best_row['aggregate_ppr']:.4f}, "
        f"agg_EHR={best_row['aggregate_ehr']:.4f}, "
        f"attempts={best_row['total_ho_attempted']}, "
        f"val_start={best_row['total_pending_validation_started']}, "
        f"val_succ={best_row['total_validation_success']}, "
        f"val_fail={best_row['total_validation_failure']}, "
        f"no_attempt_ratio={best_row['no_attempt_episode_ratio']:.3f}"
    )


if __name__ == "__main__":
    main()
