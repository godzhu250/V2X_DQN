import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_latest_tournament_folder():
    prefixes = ("Tournament_10k_", "Tournament_8000_", "Tournament_Parallel")
    if not os.path.isdir("Result"):
        return None
    folders = [
        f for f in os.listdir("Result")
        if os.path.isdir(f"Result/{f}") and f.startswith(prefixes)
    ]
    if not folders:
        return None
    return sorted(folders)[-1]


def pad_nan_and_nanmean_std(series):
    arr = np.asarray(series)

    if arr.dtype != object:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            return arr, np.zeros_like(arr, dtype=float)
        if arr.ndim == 2:
            return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
        empty = np.array([], dtype=float)
        return empty, empty

    seqs = []
    if arr.ndim <= 1:
        iterator = arr
    else:
        iterator = [arr[i] for i in range(arr.shape[0])]

    for x in iterator:
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        if x_arr.size > 0:
            seqs.append(x_arr)
    if not seqs:
        empty = np.array([], dtype=float)
        return empty, empty

    max_len = max(len(x) for x in seqs)
    padded = np.full((len(seqs), max_len), np.nan, dtype=float)
    for i, x in enumerate(seqs):
        padded[i, :len(x)] = x
    return np.nanmean(padded, axis=0), np.nanstd(padded, axis=0)


def scenario_split_from_mask(values_2d, scen_2d, target_scene):
    values_2d = np.asarray(values_2d, dtype=float)
    scen_2d = np.asarray(scen_2d)
    seqs = []
    for s in range(values_2d.shape[0]):
        mask = (scen_2d[s] == target_scene)
        seqs.append(values_2d[s][mask])
    return np.array(seqs, dtype=object)


def to_percent_scalar(v):
    v = float(v)
    return v * 100.0 if abs(v) <= 1.5 else v


def format_percent_label(val):
    val = float(val)
    if val >= 1.0:
        return f"{val:.2f}%"
    if val >= 0.01:
        return f"{val:.3f}%"
    return "<0.01%"


def _extract_seed_sequences(series):
    arr = np.asarray(series)
    seqs = []

    if arr.dtype != object:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            if arr.size > 0:
                seqs.append(arr.reshape(-1))
        elif arr.ndim >= 2:
            for i in range(arr.shape[0]):
                x = np.asarray(arr[i], dtype=float).reshape(-1)
                if x.size > 0:
                    seqs.append(x)
        return seqs

    if arr.ndim <= 1:
        iterator = arr
    else:
        iterator = [arr[i] for i in range(arr.shape[0])]

    for x in iterator:
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        if x_arr.size > 0:
            seqs.append(x_arr)
    return seqs


def mean_of_seed_means(series):
    seed_seqs = _extract_seed_sequences(series)
    if not seed_seqs:
        return np.nan

    seed_means = []
    for seq in seed_seqs:
        m = np.nanmean(np.asarray(seq, dtype=float))
        if np.isfinite(m):
            seed_means.append(float(m))

    if not seed_means:
        return np.nan
    return float(np.nanmean(seed_means))


def load_baseline_scene_kpis():
    summary_path = "Result/Stage1_Baseline/baseline_summary.csv"
    metrics_path = "Result/Stage1_Baseline/baseline_metrics.csv"

    if not os.path.exists(summary_path):
        raise ValueError("Missing baseline_summary.csv; cannot resolve baseline_threshold_dBm for scene-wise baseline KPI.")

    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        raise ValueError("baseline_summary.csv is empty; cannot resolve scene-wise baseline KPI.")

    summary_row = summary_df.iloc[-1]
    threshold = pd.to_numeric(summary_row.get("baseline_threshold_dBm", np.nan), errors="coerce")

    if not np.isfinite(threshold):
        raise ValueError("baseline_threshold_dBm is missing/invalid in baseline_summary.csv; cannot generate scene-wise bar charts reliably.")
    if not os.path.exists(metrics_path):
        raise ValueError("Missing baseline_metrics.csv; cannot generate scene-wise bar charts reliably.")

    metrics_df = pd.read_csv(metrics_path)
    if metrics_df.empty or ("threshold_dBm" not in metrics_df.columns):
        raise ValueError("baseline_metrics.csv is empty or missing threshold_dBm; cannot generate scene-wise bar charts reliably.")

    thresholds = pd.to_numeric(metrics_df["threshold_dBm"], errors="coerce").to_numpy(dtype=float)
    valid_idx = np.where(np.isfinite(thresholds))[0]
    if valid_idx.size == 0:
        raise ValueError("No valid threshold_dBm rows in baseline_metrics.csv; cannot generate scene-wise bar charts reliably.")

    close_idx = valid_idx[np.isclose(thresholds[valid_idx], float(threshold), atol=1e-8)]
    if close_idx.size > 0:
        idx = int(close_idx[0])
    else:
        raise ValueError("baseline_threshold_dBm row not found in baseline_metrics.csv; cannot generate scene-wise bar charts reliably.")

    required_scene_cols = [
        "highway_hfr", "urban_hfr",
        "highway_ppr", "urban_ppr",
        "highway_ehr", "urban_ehr",
    ]

    if not all(c in metrics_df.columns for c in required_scene_cols):
        msg = "Scene-specific baseline KPI columns not found in baseline_metrics.csv; cannot generate scene-wise bar charts reliably."
        print(msg)
        raise ValueError(msg)

    row = metrics_df.iloc[idx]
    out = {}
    for c in required_scene_cols:
        v = pd.to_numeric(row.get(c, np.nan), errors="coerce")
        if not np.isfinite(v):
            raise ValueError(f"Invalid scene-specific baseline KPI value for '{c}' at threshold {float(threshold):g}; cannot generate scene-wise bar charts reliably.")
        out[c] = float(v)
    return out


def get_model_scene_kpi(data, model_name, scene, metric):
    prefix = "h" if scene == "Highway" else "u"
    direct_key = f"{model_name}_{prefix}_{metric}"

    if direct_key in data.files:
        return mean_of_seed_means(data[direct_key])

    overall_key = f"{model_name}_{metric}"
    scen_key = f"{model_name}_scen"
    if (overall_key not in data.files) or (scen_key not in data.files):
        return np.nan

    values_2d = np.asarray(data[overall_key], dtype=float)
    scen_2d = np.asarray(data[scen_key])
    scene_series = scenario_split_from_mask(values_2d, scen_2d, scene)
    return mean_of_seed_means(scene_series)


def plot_three_bar(title, ylabel, baseline_val, mlp_val, lstm_val, filename):
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Fixed Baseline", "DQN (MLP)", "DQN (LSTM)"]
    values = [baseline_val, mlp_val, lstm_val]
    colors = ["#888888", "#1f77b4", "#ff7f0e"]

    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

    finite_vals = [v for v in values if np.isfinite(v)]
    top = max(finite_vals) if finite_vals else 1.0
    text_off = top * 0.01 if top > 0 else 0.01

    for bar, val in zip(bars, values):
        if np.isfinite(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + text_off,
                format_percent_label(val),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, max(top * 1.2, 1.0))
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    folder = find_latest_tournament_folder()
    if folder is None:
        print("No tournament folders found (expected prefixes: Tournament_10k_, Tournament_8000_, Tournament_Parallel).")
        return

    data_path = f"Result/{folder}/raw_evidence.npz"
    if not os.path.exists(data_path):
        print(f"Missing raw_evidence.npz in {folder}")
        return

    try:
        baseline_scene = load_baseline_scene_kpis()
    except ValueError:
        return
    if baseline_scene is None:
        print("Missing baseline_summary.csv or baseline KPI fields.")
        return

    data = np.load(data_path, allow_pickle=True)
    models = ["DQN (MLP)", "DQN (LSTM)"]

    scene_metric_values = {}
    for scene in ["Highway", "Urban"]:
        for metric in ["hfr", "ppr", "ehr"]:
            baseline_key = f"{scene.lower()}_{metric}"
            baseline_val = baseline_scene.get(baseline_key, np.nan)
            mlp_val = get_model_scene_kpi(data, models[0], scene, metric)
            lstm_val = get_model_scene_kpi(data, models[1], scene, metric)
            scene_metric_values[(scene, metric)] = (
                to_percent_scalar(baseline_val),
                to_percent_scalar(mlp_val),
                to_percent_scalar(lstm_val),
            )

    plots_dir = f"Result/{folder}/Plots"
    os.makedirs(plots_dir, exist_ok=True)

    plot_three_bar(
        title="Final Highway HFR Comparison",
        ylabel="Rate (%)",
        baseline_val=scene_metric_values[("Highway", "hfr")][0],
        mlp_val=scene_metric_values[("Highway", "hfr")][1],
        lstm_val=scene_metric_values[("Highway", "hfr")][2],
        filename=f"{plots_dir}/Fig16_Highway_HFR_Bar.png",
    )
    plot_three_bar(
        title="Final Urban HFR Comparison",
        ylabel="Rate (%)",
        baseline_val=scene_metric_values[("Urban", "hfr")][0],
        mlp_val=scene_metric_values[("Urban", "hfr")][1],
        lstm_val=scene_metric_values[("Urban", "hfr")][2],
        filename=f"{plots_dir}/Fig17_Urban_HFR_Bar.png",
    )
    plot_three_bar(
        title="Final Highway PPR Comparison",
        ylabel="Rate (%)",
        baseline_val=scene_metric_values[("Highway", "ppr")][0],
        mlp_val=scene_metric_values[("Highway", "ppr")][1],
        lstm_val=scene_metric_values[("Highway", "ppr")][2],
        filename=f"{plots_dir}/Fig18_Highway_PPR_Bar.png",
    )
    plot_three_bar(
        title="Final Urban PPR Comparison",
        ylabel="Rate (%)",
        baseline_val=scene_metric_values[("Urban", "ppr")][0],
        mlp_val=scene_metric_values[("Urban", "ppr")][1],
        lstm_val=scene_metric_values[("Urban", "ppr")][2],
        filename=f"{plots_dir}/Fig19_Urban_PPR_Bar.png",
    )
    plot_three_bar(
        title="Final Highway EHR Comparison",
        ylabel="Effectiveness (%)",
        baseline_val=scene_metric_values[("Highway", "ehr")][0],
        mlp_val=scene_metric_values[("Highway", "ehr")][1],
        lstm_val=scene_metric_values[("Highway", "ehr")][2],
        filename=f"{plots_dir}/Fig20_Highway_EHR_Bar.png",
    )
    plot_three_bar(
        title="Final Urban EHR Comparison",
        ylabel="Effectiveness (%)",
        baseline_val=scene_metric_values[("Urban", "ehr")][0],
        mlp_val=scene_metric_values[("Urban", "ehr")][1],
        lstm_val=scene_metric_values[("Urban", "ehr")][2],
        filename=f"{plots_dir}/Fig21_Urban_EHR_Bar.png",
    )

    print("Fig16-Fig21 scene-wise bar charts generated successfully.")


if __name__ == "__main__":
    main()
