import argparse
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
    series = pd.Series(np.asarray(x, dtype=float))
    if series.empty:
        return series.to_numpy(dtype=float)
    window = int(max(1, min(window, len(series))))
    # Keep NaN semantics: windows with no valid KPI values remain NaN.
    return series.rolling(window=window, min_periods=1).mean().to_numpy(dtype=float)


def load_model_eval_trend(train_dir):
    path = os.path.join(train_dir, "model_eval_trend.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing evaluation trend file: {path}")
    return pd.read_csv(path)


def load_baseline(baseline_dir):
    path = os.path.join(baseline_dir, "best_baseline.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing baseline file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_tail_eval_value(eval_trend_df, aliases, metric_col, tail_k=3):
    rows = pd.DataFrame()
    for alias in aliases:
        candidate = eval_trend_df[eval_trend_df["model"] == alias]
        if not candidate.empty:
            rows = candidate
            break

    if rows.empty:
        raise ValueError(
            f"No eval trend rows found for aliases={aliases} in model_eval_trend.csv"
        )

    if metric_col not in rows.columns:
        raise ValueError(
            f"Metric column '{metric_col}' not found in model_eval_trend.csv"
        )

    rows = rows.sort_values("checkpoint_episode")
    tail_rows = rows.tail(max(1, int(tail_k)))
    return float(tail_rows[metric_col].mean(skipna=True))


def plot_trend(fig_dir, file_name, title, ylabel, x_curves, y_curves, baseline_value, smooth_window):
    plt.figure(figsize=(10, 5))

    for spec in MODEL_SPECS:
        x = x_curves.get(spec["label"], np.array([]))
        curve = y_curves.get(spec["label"], np.array([]))
        if curve.size == 0:
            continue
        smooth = moving_average(curve, smooth_window)
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


def plot_metric_bar(fig_dir, file_name, title, ylabel, baseline_value, dqn_value, lstm_value):
    methods = ["Best Fixed Baseline", "DQN (MLP)", "LSTM-DQN"]

    values = [
        float(baseline_value),
        float(dqn_value),
        float(lstm_value),
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
    plt.ylabel(ylabel)
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
    eval_trend_df = load_model_eval_trend(train_dir)

    reward_curves = {}
    hfr_curves = {}
    ppr_curves = {}
    ehr_curves = {}
    x_curves = {}

    for spec in MODEL_SPECS:
        rows = pd.DataFrame()
        for alias in spec["summary_aliases"]:
            candidate = eval_trend_df[eval_trend_df["model"] == alias]
            if not candidate.empty:
                rows = candidate
                break

        if rows.empty:
            raise ValueError(
                f"No eval trend rows found for {spec['label']} in model_eval_trend.csv"
            )
        rows = rows.sort_values("checkpoint_episode")

        x_curves[spec["label"]] = rows["checkpoint_episode"].to_numpy(dtype=float)
        reward_curves[spec["label"]] = rows["reward"].to_numpy(dtype=float)
        hfr_curves[spec["label"]] = rows["hfr"].to_numpy(dtype=float)
        ppr_curves[spec["label"]] = rows["ppr"].to_numpy(dtype=float)
        ehr_curves[spec["label"]] = rows["ehr"].to_numpy(dtype=float)

    plot_trend(
        fig_dir=fig_dir,
        file_name="fig_reward_trend.png",
        title=f"{scenario_title} Reward Trend",
        ylabel="Reward",
        x_curves=x_curves,
        y_curves=reward_curves,
        baseline_value=float(baseline["mean_reward"]),
        smooth_window=args.smooth_window,
    )
    plot_trend(
        fig_dir=fig_dir,
        file_name="fig_hfr_trend.png",
        title=f"{scenario_title} HFR Trend (no-attempt episodes excluded)",
        ylabel="HFR",
        x_curves=x_curves,
        y_curves=hfr_curves,
        baseline_value=float(baseline["aggregate_hfr"]),
        smooth_window=args.smooth_window,
    )
    plot_trend(
        fig_dir=fig_dir,
        file_name="fig_ppr_trend.png",
        title=f"{scenario_title} PPR Trend (no-attempt episodes excluded)",
        ylabel="PPR",
        x_curves=x_curves,
        y_curves=ppr_curves,
        baseline_value=float(baseline["aggregate_ppr"]),
        smooth_window=args.smooth_window,
    )
    plot_trend(
        fig_dir=fig_dir,
        file_name="fig_ehr_trend.png",
        title=f"{scenario_title} EHR Trend (no-attempt episodes excluded)",
        ylabel="EHR",
        x_curves=x_curves,
        y_curves=ehr_curves,
        baseline_value=float(baseline["aggregate_ehr"]),
        smooth_window=args.smooth_window,
    )

    plot_metric_bar(
        fig_dir=fig_dir,
        file_name="fig_hfr_bar.png",
        title=f"{scenario_title} Late-Stage HFR Comparison",
        ylabel="HFR",
        baseline_value=float(baseline["aggregate_hfr"]),
        dqn_value=get_tail_eval_value(eval_trend_df, ["DQN (MLP)"], "hfr", tail_k=3),
        lstm_value=get_tail_eval_value(eval_trend_df, ["LSTM-DQN", "DRQN (LSTM)"], "hfr", tail_k=3),
    )
    plot_metric_bar(
        fig_dir=fig_dir,
        file_name="fig_ppr_bar.png",
        title=f"{scenario_title} Late-Stage PPR Comparison",
        ylabel="PPR",
        baseline_value=float(baseline["aggregate_ppr"]),
        dqn_value=get_tail_eval_value(eval_trend_df, ["DQN (MLP)"], "ppr", tail_k=3),
        lstm_value=get_tail_eval_value(eval_trend_df, ["LSTM-DQN", "DRQN (LSTM)"], "ppr", tail_k=3),
    )
    plot_metric_bar(
        fig_dir=fig_dir,
        file_name="fig_ehr_bar.png",
        title=f"{scenario_title} Late-Stage EHR Comparison",
        ylabel="EHR",
        baseline_value=float(baseline["aggregate_ehr"]),
        dqn_value=get_tail_eval_value(eval_trend_df, ["DQN (MLP)"], "ehr", tail_k=3),
        lstm_value=get_tail_eval_value(eval_trend_df, ["LSTM-DQN", "DRQN (LSTM)"], "ehr", tail_k=3),
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
