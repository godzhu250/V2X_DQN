import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def rolling_mean(x, window=200):
    return pd.Series(x, dtype=float).rolling(window, min_periods=1).mean().values


def smooth_series(arr, window=200):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return arr
    window = int(max(1, min(window, len(arr))))
    return rolling_mean(arr, window=window)


def plot_and_save(data_dict, title, ylabel, filename, folder, smooth=200, ylim=None, ylim_bottom=None,
                  metric_type=None, baseline_value=None):
    plt.figure(figsize=(10, 5))
    y_max_candidates = []
    for name, series in data_dict.items():
        mean_curve, _ = pad_nan_and_nanmean_std(series)
        if len(mean_curve) == 0:
            continue
        mean_s = smooth_series(mean_curve, window=smooth)
        if len(mean_s) == 0:
            continue
        y_max_candidates.append(float(np.nanmax(mean_s)))
        plt.plot(mean_s, label=name, linewidth=2)

    if baseline_value is not None and np.isfinite(baseline_value):
        plt.axhline(float(baseline_value), color="gray", linestyle="--", linewidth=1.8, label="Fixed Baseline")

    max_mean = max(y_max_candidates) if y_max_candidates else 0.0
    if metric_type == "HFR":
        plt.ylim(0.0, max(0.02, 1.2 * max_mean))
    elif metric_type == "PPR":
        plt.ylim(0.0, min(1.0, max(0.3, 1.2 * max_mean)))
    elif metric_type == "EHR":
        plt.ylim(0.0, 1.05)
    elif ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    elif ylim_bottom is not None:
        plt.ylim(bottom=ylim_bottom)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    outdir = f"Result/{folder}/Plots"
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{filename}", dpi=300)
    plt.close()


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
    for x in arr:
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


def get_scene_reward_series(data, model_name, scene):
    prefix = "h" if scene == "Highway" else "u"
    key_r = f"{model_name}_{prefix}_r"
    if key_r in data.files:
        return data[key_r]
    r2d = np.asarray(data[f"{model_name}_r"], dtype=float)
    scen2d = np.asarray(data[f"{model_name}_scen"])
    return scenario_split_from_mask(r2d, scen2d, scene)


def plot_fig6_dual_scene(data, models, folder, highway_baseline, urban_baseline, smooth=200):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    model_color = {
        "DQN (MLP)": "tab:blue",
        "DQN (LSTM)": "tab:orange",
    }

    scen_cfg = [
        ("Highway", highway_baseline, axes[0], "Highway Reward Gain over Static Baseline"),
        ("Urban", urban_baseline, axes[1], "Urban Reward Gain over Static Baseline"),
    ]
    for scene, baseline, ax, title in scen_cfg:
        for m in models:
            reward_series = get_scene_reward_series(data, m, scene)
            mean_curve, _ = pad_nan_and_nanmean_std(reward_series)
            if len(mean_curve) == 0:
                continue
            gain_curve = mean_curve - float(baseline)
            gain_s = smooth_series(gain_curve, window=smooth)
            if len(gain_s) == 0:
                continue
            ax.plot(gain_s, linewidth=2, color=model_color.get(m, None), label=m)

        ax.axhline(
            0.0,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label="Zero-gain line (equal to fixed baseline)"
        )
        ax.set_title(title)
        ax.set_ylabel("Reward Gain")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

    axes[0].set_xlabel("Episode Index")
    axes[1].set_xlabel("Episode Index")
    plt.tight_layout()
    outdir = f"Result/{folder}/Plots"
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f"{outdir}/Fig6_Reward.png", dpi=300)
    plt.close(fig)


def scenario_split_from_mask(values_2d, scen_2d, target_scene):
    values_2d = np.asarray(values_2d, dtype=float)
    scen_2d = np.asarray(scen_2d)
    seqs = []
    for s in range(values_2d.shape[0]):
        mask = (scen_2d[s] == target_scene)
        seqs.append(values_2d[s][mask])
    return np.array(seqs, dtype=object)


def load_baseline_kpis():
    path = "Result/Stage1_Baseline/baseline_summary.csv"
    if not os.path.exists(path):
        return {}

    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    if df.empty:
        return {}

    row = df.iloc[-1]
    out = {}
    for key in ["baseline_hfr", "baseline_ppr", "baseline_ehr"]:
        if key in df.columns:
            val = pd.to_numeric(row[key], errors="coerce")
            if np.isfinite(val):
                out[key] = float(val)
    return out


def load_baseline_reward_by_sweet_spot():
    summary_path = "Result/Stage1_Baseline/baseline_summary.csv"
    metrics_path = "Result/Stage1_Baseline/baseline_metrics.csv"
    if (not os.path.exists(summary_path)) or (not os.path.exists(metrics_path)):
        return None, None

    try:
        summary_df = pd.read_csv(summary_path)
        metrics_df = pd.read_csv(metrics_path)
    except Exception:
        return None, None

    if summary_df.empty:
        return None, None
    if "baseline_threshold_dBm" not in summary_df.columns:
        return None, None
    required_cols = {"threshold_dBm", "highway_score", "urban_score"}
    if not required_cols.issubset(set(metrics_df.columns)):
        return None, None

    threshold = pd.to_numeric(summary_df.iloc[-1]["baseline_threshold_dBm"], errors="coerce")
    if not np.isfinite(threshold):
        return None, None

    thresholds = pd.to_numeric(metrics_df["threshold_dBm"], errors="coerce").to_numpy(dtype=float)
    valid_idx = np.where(np.isfinite(thresholds))[0]
    if valid_idx.size == 0:
        return None, None

    close_idx = valid_idx[np.isclose(thresholds[valid_idx], float(threshold), atol=1e-8)]
    if close_idx.size > 0:
        idx = int(close_idx[0])
    else:
        idx = int(valid_idx[np.argmin(np.abs(thresholds[valid_idx] - float(threshold)))])

    h_base = pd.to_numeric(metrics_df.iloc[idx]["highway_score"], errors="coerce")
    u_base = pd.to_numeric(metrics_df.iloc[idx]["urban_score"], errors="coerce")
    if (not np.isfinite(h_base)) or (not np.isfinite(u_base)):
        return None, None
    return float(h_base), float(u_base)


def block_mean_per_seed(obj_series, block_size=200):
    arr = np.asarray(obj_series, dtype=object)
    block_seqs = []
    for seq in arr:
        x = np.asarray(seq, dtype=float).reshape(-1)
        if x.size == 0:
            continue
        n_blocks = int(np.ceil(x.size / block_size))
        block_vals = []
        for i in range(n_blocks):
            seg = x[i * block_size:(i + 1) * block_size]
            if seg.size > 0:
                block_vals.append(float(np.mean(seg)))
        if block_vals:
            block_seqs.append(np.asarray(block_vals, dtype=float))
    return np.array(block_seqs, dtype=object)


def cumulative_stage_mean_per_seed(series, stage_size=500):
    arr = np.asarray(series)
    seqs = []
    if arr.dtype == object:
        for x in arr:
            x_arr = np.asarray(x, dtype=float).reshape(-1)
            if x_arr.size > 0:
                seqs.append(x_arr)
    else:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            if arr.size > 0:
                seqs.append(arr.reshape(-1))
        elif arr.ndim == 2:
            for i in range(arr.shape[0]):
                x_arr = np.asarray(arr[i], dtype=float).reshape(-1)
                if x_arr.size > 0:
                    seqs.append(x_arr)

    out = []
    for seq in seqs:
        n_stages = int(np.ceil(seq.size / stage_size))
        stage_vals = []
        for k in range(n_stages):
            end = min((k + 1) * stage_size, seq.size)
            if end > 0:
                stage_vals.append(float(np.mean(seq[:end])))
        if stage_vals:
            out.append(np.asarray(stage_vals, dtype=float))
    return np.array(out, dtype=object)


def stage_mean_per_seed(series, stage_size=500):
    arr = np.asarray(series)
    seqs = []
    if arr.dtype == object:
        for x in arr:
            x_arr = np.asarray(x, dtype=float).reshape(-1)
            if x_arr.size > 0:
                seqs.append(x_arr)
    else:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            if arr.size > 0:
                seqs.append(arr.reshape(-1))
        elif arr.ndim == 2:
            for i in range(arr.shape[0]):
                x_arr = np.asarray(arr[i], dtype=float).reshape(-1)
                if x_arr.size > 0:
                    seqs.append(x_arr)

    out = []
    for seq in seqs:
        n_stages = int(np.ceil(seq.size / stage_size))
        stage_vals = []
        for k in range(n_stages):
            start = k * stage_size
            end = min((k + 1) * stage_size, seq.size)
            if end > start:
                stage_vals.append(float(np.mean(seq[start:end])))
        if stage_vals:
            out.append(np.asarray(stage_vals, dtype=float))
    return np.array(out, dtype=object)


def plot_stagewise_cumulative_kpi(data, models, folder, key_suffix, title, ylabel, filename, baseline_value=None, stage_size=500):
    plt.figure(figsize=(10, 5))
    for m in models:
        key = f"{m}_{key_suffix}"
        if key not in data.files:
            continue
        stage_series = cumulative_stage_mean_per_seed(data[key], stage_size=stage_size)
        mean_curve, _ = pad_nan_and_nanmean_std(stage_series)
        if len(mean_curve) == 0:
            continue
        x = np.arange(1, len(mean_curve) + 1)
        plt.plot(x, mean_curve, linewidth=2, label=m)

    if baseline_value is not None and np.isfinite(baseline_value):
        plt.axhline(float(baseline_value), color="gray", linestyle="--", linewidth=1.8, label="Fixed Baseline")

    plt.title(title)
    plt.xlabel("Training Stage (each stage = 500 episodes)")
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    outdir = f"Result/{folder}/Plots"
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{filename}", dpi=300)
    plt.close()


def plot_stagewise_mean_kpi(data, models, folder, key_suffix, title, ylabel, filename, baseline_value=None, stage_size=500):
    plt.figure(figsize=(10, 5))
    for m in models:
        key = f"{m}_{key_suffix}"
        if key not in data.files:
            continue
        stage_series = stage_mean_per_seed(data[key], stage_size=stage_size)
        mean_curve, _ = pad_nan_and_nanmean_std(stage_series)
        if len(mean_curve) == 0:
            continue
        x = np.arange(1, len(mean_curve) + 1)
        plt.plot(x, mean_curve, linewidth=2, label=m)

    if baseline_value is not None and np.isfinite(baseline_value):
        plt.axhline(float(baseline_value), color="gray", linestyle="--", linewidth=1.8, label="Fixed Baseline")

    plt.title(title)
    plt.xlabel("Training Stage (each stage = 500 episodes)")
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    outdir = f"Result/{folder}/Plots"
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{filename}", dpi=300)
    plt.close()


def plot_fig9_reward_gain_blockwise(data, models, folder, highway_baseline, urban_baseline, block_size=200):
    for scene, prefix, baseline in [
        ("Highway", "h", highway_baseline),
        ("Urban", "u", urban_baseline),
    ]:
        plt.figure(figsize=(10, 5))
        for m in models:
            key_r = f"{m}_{prefix}_r"
            if key_r in data.files:
                scen_reward = data[key_r]
            else:
                r2d = np.asarray(data[f"{m}_r"], dtype=float)
                scen2d = np.asarray(data[f"{m}_scen"])
                scen_reward = scenario_split_from_mask(r2d, scen2d, scene)

            block_series = block_mean_per_seed(scen_reward, block_size=block_size)
            mean_curve, _ = pad_nan_and_nanmean_std(block_series)
            if len(mean_curve) == 0:
                continue
            gain_curve = mean_curve - float(baseline)
            gain_s = rolling_mean(gain_curve, window=3)
            plt.plot(gain_s, linewidth=2, label=m)

        plt.title(f"Block-wise {scene} Reward Gain over Static Baseline")
        plt.xlabel("Block Index (each block = 200 episodes)")
        plt.ylabel("Reward Gain vs. Static Baseline")
        plt.axhline(
            0.0,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label="Zero-gain line (equal to fixed baseline)"
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        outdir = f"Result/{folder}/Plots"
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f"{outdir}/Fig9_{scene}_Reward.png", dpi=300)
        plt.close()


def plot_fig12_ehr_gain(data, models, folder, baseline_ehr, smooth=200):
    for scene, prefix in [("Highway", "h"), ("Urban", "u")]:
        plt.figure(figsize=(10, 5))
        for m in models:
            key_ehr = f"{m}_{prefix}_ehr"
            if key_ehr in data.files:
                scen_ehr = data[key_ehr]
            else:
                ehr2d = np.asarray(data[f"{m}_ehr"], dtype=float)
                scen2d = np.asarray(data[f"{m}_scen"])
                scen_ehr = scenario_split_from_mask(ehr2d, scen2d, scene)

            mean_curve, _ = pad_nan_and_nanmean_std(scen_ehr)
            if len(mean_curve) == 0:
                continue
            gain_curve = mean_curve - float(baseline_ehr)
            gain_s = smooth_series(gain_curve, window=smooth)
            if len(gain_s) == 0:
                continue
            x = np.arange(len(gain_s), dtype=int)
            plt.plot(x, gain_s, linewidth=2, label=m)

        plt.axhline(
            0.0,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label="Zero-gain line (equal to fixed baseline)"
        )
        plt.title(f"{scene} EHR Gain over Baseline")
        plt.xlabel("Episode Index")
        plt.ylabel("EHR Gain")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        outdir = f"Result/{folder}/Plots"
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f"{outdir}/Fig12_{scene}_EHR.png", dpi=300)
        plt.close()


def main():
    folders = [f for f in os.listdir("Result")
               if f.startswith("Tournament_10k_") or f.startswith("Tournament_8000_") or f.startswith("Tournament_Parallel")]
    if not folders:
        print("No tournament folders found.")
        return
    folder = sorted(folders)[-1]

    data_path = f"Result/{folder}/raw_evidence.npz"
    if not os.path.exists(data_path):
        print(f"Missing raw_evidence.npz in {folder}")
        return

    data = np.load(data_path, allow_pickle=True)
    models = ["DQN (MLP)", "DQN (LSTM)"]

    print(f"Generating Fig 6-12 for {folder}...")
    baseline_kpis = load_baseline_kpis()
    highway_baseline, urban_baseline = load_baseline_reward_by_sweet_spot()

    if (highway_baseline is None) or (urban_baseline is None):
        print("Missing Stage1 baseline reward for Fig6/Fig9 reward gain; skip Fig6/Fig9.")
    else:
        plot_fig6_dual_scene(
            data,
            models,
            folder,
            highway_baseline=highway_baseline,
            urban_baseline=urban_baseline,
            smooth=200,
        )
    plot_and_save(
        {m: data[f"{m}_hfr"] for m in models},
        "Handover Failure Rate (HFR) Trend", "Rate", "Fig7a_HFR.png", folder,
        smooth=200, metric_type="HFR", baseline_value=baseline_kpis.get("baseline_hfr")
    )
    plot_and_save(
        {m: data[f"{m}_ppr"] for m in models},
        "Ping-pong Rate (PPR) Trend", "Rate", "Fig7b_PPR.png", folder,
        smooth=200, metric_type="PPR", baseline_value=baseline_kpis.get("baseline_ppr")
    )
    plot_and_save(
        {m: data[f"{m}_ehr"] for m in models},
        "Effective Handover Rate (EHR) Trend", "Percentage", "Fig8_EHR.png", folder,
        smooth=200, metric_type="EHR", baseline_value=baseline_kpis.get("baseline_ehr")
    )

    plot_stagewise_cumulative_kpi(
        data=data,
        models=models,
        folder=folder,
        key_suffix="hfr",
        title="Stage-wise Cumulative HFR Evolution",
        ylabel="Rate",
        filename="Fig7c_HFR_Cumulative.png",
        baseline_value=baseline_kpis.get("baseline_hfr"),
        stage_size=500,
    )
    plot_stagewise_cumulative_kpi(
        data=data,
        models=models,
        folder=folder,
        key_suffix="ppr",
        title="Stage-wise Cumulative PPR Evolution",
        ylabel="Rate",
        filename="Fig7d_PPR_Cumulative.png",
        baseline_value=baseline_kpis.get("baseline_ppr"),
        stage_size=500,
    )
    plot_stagewise_cumulative_kpi(
        data=data,
        models=models,
        folder=folder,
        key_suffix="ehr",
        title="Stage-wise Cumulative EHR Evolution",
        ylabel="Percentage",
        filename="Fig8b_EHR_Cumulative.png",
        baseline_value=baseline_kpis.get("baseline_ehr"),
        stage_size=500,
    )

    plot_stagewise_mean_kpi(
        data=data,
        models=models,
        folder=folder,
        key_suffix="hfr",
        title="Stage-wise Mean HFR by Training Stage",
        ylabel="Rate",
        filename="Fig7e_HFR_StageMean.png",
        baseline_value=baseline_kpis.get("baseline_hfr"),
        stage_size=500,
    )
    plot_stagewise_mean_kpi(
        data=data,
        models=models,
        folder=folder,
        key_suffix="ppr",
        title="Stage-wise Mean PPR by Training Stage",
        ylabel="Rate",
        filename="Fig7f_PPR_StageMean.png",
        baseline_value=baseline_kpis.get("baseline_ppr"),
        stage_size=500,
    )
    plot_stagewise_mean_kpi(
        data=data,
        models=models,
        folder=folder,
        key_suffix="ehr",
        title="Stage-wise Mean EHR by Training Stage",
        ylabel="Percentage",
        filename="Fig8c_EHR_StageMean.png",
        baseline_value=baseline_kpis.get("baseline_ehr"),
        stage_size=500,
    )

    if (highway_baseline is not None) and (urban_baseline is not None):
        plot_fig9_reward_gain_blockwise(
            data,
            models,
            folder,
            highway_baseline=highway_baseline,
            urban_baseline=urban_baseline,
            block_size=200,
        )

    baseline_ehr = baseline_kpis.get("baseline_ehr")
    if baseline_ehr is None or (not np.isfinite(float(baseline_ehr))):
        print("Missing baseline_ehr for Fig12 EHR gain; skip Fig12.")
    else:
        plot_fig12_ehr_gain(data, models, folder, baseline_ehr=float(baseline_ehr), smooth=200)

    print("Fig 6-12 Trends Generated successfully.")


if __name__ == "__main__":
    main()
