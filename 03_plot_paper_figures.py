import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_SPECS = [
    ("DQN (MLP)", "DQN_MLP", "tab:blue"),
    ("DRQN (LSTM)", "DRQN_LSTM", "tab:orange"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        default="highway",
        choices=["highway", "urban"],
        help="Scenario folder under Result/",
    )
    parser.add_argument("--smooth-window", type=int, default=100)
    return parser.parse_args()


def moving_average(x, window):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    window = int(max(1, min(window, x.size)))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")


def load_curves(train_dir, model_tag, metric):
    paths = sorted(glob.glob(os.path.join(train_dir, f"{model_tag}_seed*.csv")))
    if not paths:
        return np.array([]), []

    curves = []
    for p in paths:
        df = pd.read_csv(p)
        if metric not in df.columns:
            continue
        curves.append(df[metric].to_numpy(dtype=float))

    if not curves:
        return np.array([]), paths

    min_len = min(len(c) for c in curves)
    arr = np.array([c[:min_len] for c in curves], dtype=float)
    mean_curve = np.mean(arr, axis=0)
    return mean_curve, paths


def load_baseline(baseline_dir):
    path = os.path.join(baseline_dir, "best_baseline.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing baseline file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_summary(train_dir):
    path = os.path.join(train_dir, "model_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model summary file: {path}")
    return pd.read_csv(path)


def plot_reward_curve(fig_dir, scenario_title, baseline, reward_curves, smooth_window):
    plt.figure(figsize=(10, 5))

    for model_name, _, color in MODEL_SPECS:
        curve = reward_curves.get(model_name, np.array([]))
        if curve.size == 0:
            continue
        smooth = moving_average(curve, smooth_window)

        x = np.arange(1, len(curve) + 1)
        plt.plot(x, curve, color=color, alpha=0.20, linewidth=1.0)
        plt.plot(x, smooth, color=color, linewidth=2.0, label=model_name)

    baseline_reward = float(baseline["mean_reward"])
    plt.axhline(
        baseline_reward,
        color="gray",
        linestyle="--",
        linewidth=1.8,
        label="Best Fixed Baseline",
    )

    plt.title(f"{scenario_title}: Reward Curve (Absolute)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig1_reward_curve.png"), dpi=300)
    plt.close()


def plot_hfr_ppr(fig_dir, scenario_title, baseline, hfr_curves, ppr_curves, smooth_window):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for model_name, _, color in MODEL_SPECS:
        hfr = hfr_curves.get(model_name, np.array([]))
        ppr = ppr_curves.get(model_name, np.array([]))

        if hfr.size > 0:
            axes[0].plot(np.arange(1, len(hfr) + 1), moving_average(hfr, smooth_window), color=color, linewidth=2.0, label=model_name)
        if ppr.size > 0:
            axes[1].plot(np.arange(1, len(ppr) + 1), moving_average(ppr, smooth_window), color=color, linewidth=2.0, label=model_name)

    axes[0].axhline(float(baseline["aggregate_hfr"]), color="gray", linestyle="--", linewidth=1.6, label="Best Fixed Baseline")
    axes[1].axhline(float(baseline["aggregate_ppr"]), color="gray", linestyle="--", linewidth=1.6, label="Best Fixed Baseline")

    axes[0].set_title(f"{scenario_title}: HFR Trend")
    axes[1].set_title(f"{scenario_title}: PPR Trend")
    axes[0].set_ylabel("HFR")
    axes[1].set_ylabel("PPR")
    axes[1].set_xlabel("Episode")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig2_hfr_ppr_trend.png"), dpi=300)
    plt.close(fig)


def plot_ehr(fig_dir, scenario_title, baseline, ehr_curves, smooth_window):
    plt.figure(figsize=(10, 5))

    for model_name, _, color in MODEL_SPECS:
        curve = ehr_curves.get(model_name, np.array([]))
        if curve.size == 0:
            continue
        plt.plot(
            np.arange(1, len(curve) + 1),
            moving_average(curve, smooth_window),
            color=color,
            linewidth=2.0,
            label=model_name,
        )

    plt.axhline(
        float(baseline["aggregate_ehr"]),
        color="gray",
        linestyle="--",
        linewidth=1.8,
        label="Best Fixed Baseline",
    )

    plt.title(f"{scenario_title}: EHR Trend")
    plt.xlabel("Episode")
    plt.ylabel("EHR")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig3_ehr_trend.png"), dpi=300)
    plt.close()


def _get_model_row(model_summary_df, model_name):
    rows = model_summary_df[model_summary_df["model"] == model_name]
    if rows.empty:
        raise ValueError(f"Model '{model_name}' not found in model_summary.csv")
    return rows.iloc[0]


def plot_final_bar(fig_dir, baseline, model_summary_df, scenario_title):
    methods = ["Best Fixed Baseline", "DQN (MLP)", "DRQN (LSTM)"]
    metrics = ["HFR", "PPR", "EHR", "Reward"]

    dqn_row = _get_model_row(model_summary_df, "DQN (MLP)")
    drqn_row = _get_model_row(model_summary_df, "DRQN (LSTM)")

    values = {
        "Best Fixed Baseline": [
            float(baseline["aggregate_hfr"]),
            float(baseline["aggregate_ppr"]),
            float(baseline["aggregate_ehr"]),
            float(baseline["mean_reward"]),
        ],
        "DQN (MLP)": [
            float(dqn_row["aggregate_hfr"]),
            float(dqn_row["aggregate_ppr"]),
            float(dqn_row["aggregate_ehr"]),
            float(dqn_row["mean_reward"]),
        ],
        "DRQN (LSTM)": [
            float(drqn_row["aggregate_hfr"]),
            float(drqn_row["aggregate_ppr"]),
            float(drqn_row["aggregate_ehr"]),
            float(drqn_row["mean_reward"]),
        ],
    }

    x = np.arange(len(metrics), dtype=float)
    width = 0.25

    plt.figure(figsize=(10, 5))
    colors = ["#888888", "tab:blue", "tab:orange"]

    for i, method in enumerate(methods):
        y = values[method]
        plt.bar(x + (i - 1) * width, y, width=width, label=method, color=colors[i], edgecolor="black", linewidth=0.6)

    plt.xticks(x, metrics)
    plt.title(f"{scenario_title}: Final KPI/Reward Comparison")
    plt.ylabel("Value")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig4_final_bar_chart.png"), dpi=300)
    plt.close()


def main():
    args = parse_args()

    scenario_folder = args.scenario.lower()
    scenario_title = "Highway" if scenario_folder == "highway" else "Urban"

    base_dir = os.path.join("Result", scenario_folder)
    baseline_dir = os.path.join(base_dir, "baseline")
    train_dir = os.path.join(base_dir, "train")
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    baseline = load_baseline(baseline_dir)
    model_summary_df = load_model_summary(train_dir)

    reward_curves = {}
    hfr_curves = {}
    ppr_curves = {}
    ehr_curves = {}

    for model_name, model_tag, _ in MODEL_SPECS:
        reward_curve, paths = load_curves(train_dir, model_tag, "reward")
        hfr_curve, _ = load_curves(train_dir, model_tag, "hfr")
        ppr_curve, _ = load_curves(train_dir, model_tag, "ppr")
        ehr_curve, _ = load_curves(train_dir, model_tag, "ehr")

        if len(paths) == 0:
            raise FileNotFoundError(
                f"No per-seed training files found for {model_name} in {train_dir}. "
                f"Expected files like {model_tag}_seed*.csv"
            )

        reward_curves[model_name] = reward_curve
        hfr_curves[model_name] = hfr_curve
        ppr_curves[model_name] = ppr_curve
        ehr_curves[model_name] = ehr_curve

    plot_reward_curve(fig_dir, scenario_title, baseline, reward_curves, args.smooth_window)
    plot_hfr_ppr(fig_dir, scenario_title, baseline, hfr_curves, ppr_curves, args.smooth_window)
    plot_ehr(fig_dir, scenario_title, baseline, ehr_curves, args.smooth_window)
    plot_final_bar(fig_dir, baseline, model_summary_df, scenario_title)

    print(f"Saved figures to: {fig_dir}")
    print("- fig1_reward_curve.png")
    print("- fig2_hfr_ppr_trend.png")
    print("- fig3_ehr_trend.png")
    print("- fig4_final_bar_chart.png")


if __name__ == "__main__":
    main()
