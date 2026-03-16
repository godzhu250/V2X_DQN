import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_SPECS = [
    {
        "label": "DQN (MLP)",
        "summary_aliases": ["DQN (MLP)"],
        "file_tags": ["DQN_MLP"],
        "color": "red",
    },
    {
        "label": "LSTM-DQN",
        "summary_aliases": ["LSTM-DQN", "DRQN (LSTM)"],
        "file_tags": ["LSTM_DQN", "DRQN_LSTM"],
        "color": "blue",
    },
]

BASELINE_COLOR = "gray"


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


def _find_files(train_dir, file_tags):
    for tag in file_tags:
        paths = sorted(glob.glob(os.path.join(train_dir, f"{tag}_seed*.csv")))
        if paths:
            return paths
    return []


def load_mean_curve(train_dir, file_tags, metric):
    paths = _find_files(train_dir, file_tags)
    if not paths:
        return np.array([]), []

    curves = []
    for path in paths:
        df = pd.read_csv(path)
        if metric not in df.columns:
            continue
        curves.append(df[metric].to_numpy(dtype=float))

    if not curves:
        return np.array([]), paths

    min_len = min(len(c) for c in curves)
    arr = np.array([c[:min_len] for c in curves], dtype=float)
    return np.mean(arr, axis=0), paths


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


def _get_model_row(model_summary_df, aliases):
    for alias in aliases:
        rows = model_summary_df[model_summary_df["model"] == alias]
        if not rows.empty:
            return rows.iloc[0]
    raise ValueError(f"No model row found for aliases: {aliases}")


def plot_trend(fig_dir, file_name, title, ylabel, curves, baseline_value, smooth_window):
    plt.figure(figsize=(10, 5))

    for spec in MODEL_SPECS:
        curve = curves.get(spec["label"], np.array([]))
        if curve.size == 0:
            continue
        smooth = moving_average(curve, smooth_window)
        x = np.arange(1, len(curve) + 1)
        plt.plot(x, smooth, color=spec["color"], linewidth=2.0, label=spec["label"])

    plt.axhline(
        float(baseline_value),
        color=BASELINE_COLOR,
        linestyle="--",
        linewidth=1.8,
        label="Fixed Baseline",
    )

    plt.title(title)
    plt.xlabel("Episode Index")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, file_name), dpi=300)
    plt.close()


def plot_metric_bar(fig_dir, file_name, title, metric_name, baseline_value, model_summary_df):
    methods = ["Best Fixed Baseline", "DQN (MLP)", "LSTM-DQN"]

    dqn_row = _get_model_row(model_summary_df, ["DQN (MLP)"])
    lstm_row = _get_model_row(model_summary_df, ["LSTM-DQN", "DRQN (LSTM)"])

    values = [
        float(baseline_value),
        float(dqn_row[metric_name]),
        float(lstm_row[metric_name]),
    ]

    colors = ["#888888", "red", "blue"]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(methods, values, color=colors, edgecolor="black", linewidth=0.6)

    offset = max(values) * 0.01 if max(values) > 0 else 0.01
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.title(title)
    plt.ylabel(metric_name.replace("aggregate_", "").upper())
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, file_name), dpi=300)
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

    for spec in MODEL_SPECS:
        reward_curve, paths = load_mean_curve(train_dir, spec["file_tags"], "reward")
        hfr_curve, _ = load_mean_curve(train_dir, spec["file_tags"], "hfr")
        ppr_curve, _ = load_mean_curve(train_dir, spec["file_tags"], "ppr")
        ehr_curve, _ = load_mean_curve(train_dir, spec["file_tags"], "ehr")

        if not paths:
            raise FileNotFoundError(
                f"No per-seed training files found for {spec['label']} in {train_dir}. "
                f"Expected tags: {spec['file_tags']}"
            )

        reward_curves[spec["label"]] = reward_curve
        hfr_curves[spec["label"]] = hfr_curve
        ppr_curves[spec["label"]] = ppr_curve
        ehr_curves[spec["label"]] = ehr_curve

    plot_trend(
        fig_dir=fig_dir,
        file_name="fig_reward_trend.png",
        title=f"{scenario_title} Reward Trend",
        ylabel="Reward",
        curves=reward_curves,
        baseline_value=float(baseline["mean_reward"]),
        smooth_window=args.smooth_window,
    )
    plot_trend(
        fig_dir=fig_dir,
        file_name="fig_hfr_trend.png",
        title=f"{scenario_title} HFR Trend",
        ylabel="HFR",
        curves=hfr_curves,
        baseline_value=float(baseline["aggregate_hfr"]),
        smooth_window=args.smooth_window,
    )
    plot_trend(
        fig_dir=fig_dir,
        file_name="fig_ppr_trend.png",
        title=f"{scenario_title} PPR Trend",
        ylabel="PPR",
        curves=ppr_curves,
        baseline_value=float(baseline["aggregate_ppr"]),
        smooth_window=args.smooth_window,
    )
    plot_trend(
        fig_dir=fig_dir,
        file_name="fig_ehr_trend.png",
        title=f"{scenario_title} EHR Trend",
        ylabel="EHR",
        curves=ehr_curves,
        baseline_value=float(baseline["aggregate_ehr"]),
        smooth_window=args.smooth_window,
    )

    plot_metric_bar(
        fig_dir=fig_dir,
        file_name="fig_hfr_bar.png",
        title=f"{scenario_title} Final HFR Comparison",
        metric_name="aggregate_hfr",
        baseline_value=float(baseline["aggregate_hfr"]),
        model_summary_df=model_summary_df,
    )
    plot_metric_bar(
        fig_dir=fig_dir,
        file_name="fig_ppr_bar.png",
        title=f"{scenario_title} Final PPR Comparison",
        metric_name="aggregate_ppr",
        baseline_value=float(baseline["aggregate_ppr"]),
        model_summary_df=model_summary_df,
    )
    plot_metric_bar(
        fig_dir=fig_dir,
        file_name="fig_ehr_bar.png",
        title=f"{scenario_title} Final EHR Comparison",
        metric_name="aggregate_ehr",
        baseline_value=float(baseline["aggregate_ehr"]),
        model_summary_df=model_summary_df,
    )

    print(f"Saved figures to: {fig_dir}")
    print("- fig_reward_trend.png")
    print("- fig_hfr_trend.png")
    print("- fig_ppr_trend.png")
    print("- fig_ehr_trend.png")
    print("- fig_hfr_bar.png")
    print("- fig_ppr_bar.png")
    print("- fig_ehr_bar.png")


if __name__ == "__main__":
    main()
