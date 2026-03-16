import os
import numpy as np
import matplotlib.pyplot as plt


class RefStyleConvergencePlotter:
    def __init__(self, save_dir, roster):
        self.save_dir = save_dir
        self.roster = list(roster)
        self.color_map = {
            "DQN (MLP)": "red",
            "DQN (LSTM)": "blue",
        }

    def plot_refstyle_convergence_with_stage_ema(self, evidence):
        def safe_seed_mean(arr):
            arr = np.asarray(arr)
            if arr.size == 0:
                return np.array([], dtype=float)

            if arr.dtype == object:
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
                    return np.array([], dtype=float)
                max_len = max(len(x) for x in seqs)
                padded = np.full((len(seqs), max_len), np.nan, dtype=float)
                for i, seq in enumerate(seqs):
                    padded[i, :len(seq)] = seq
                return np.nanmean(padded, axis=0)

            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 1:
                return arr
            if arr.ndim >= 2:
                return np.nanmean(arr, axis=0)
            return np.array([], dtype=float)

        def block_mean(x, block_size=400):
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size == 0:
                return np.array([], dtype=float)
            n_full = int(x.size // block_size)
            if n_full <= 0:
                return np.array([], dtype=float)
            trimmed = x[: n_full * block_size].reshape(n_full, block_size)
            return np.nanmean(trimmed, axis=1)

        def ema_smooth(data, alpha=0.35):
            data = np.asarray(data, dtype=float).reshape(-1)
            if data.size == 0:
                return np.array([], dtype=float)
            out = np.empty_like(data, dtype=float)
            out[0] = data[0]
            for i in range(1, data.size):
                out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
            return out

        def make_stage2_ema_curve(y_raw, block_size=400, alpha=0.35):
            y_stage = block_mean(y_raw, block_size=block_size)
            n_stage = int(y_stage.size)
            if n_stage <= 1:
                return np.array([], dtype=float), np.array([], dtype=float), n_stage
            y_stage = y_stage[1:]  # stage 2 onwards
            y_plot = ema_smooth(y_stage, alpha=alpha)
            x_plot = np.arange(2, 2 + y_plot.size, dtype=int)
            return x_plot, y_plot, n_stage

        os.makedirs(self.save_dir, exist_ok=True)
        files = list(evidence.files)

        # Fig26: Reward
        print("[Fig26] Ref-style convergence starts from stage 2 for statistical stability.")
        plt.figure(figsize=(10, 5))
        fig26_has_curve = False
        for name in self.roster:
            reward_key = None
            for k in [f"{name}_reward", f"{name}_r", f"{name}_overall_reward", f"{name}_rewards"]:
                if k in evidence.files:
                    reward_key = k
                    break
            if reward_key is None:
                print(f"[Warn][Fig26][{name}] reward key missing. available keys: {files}")
                continue

            y_raw = safe_seed_mean(evidence[reward_key])
            x_plot, y_plot, n_stage = make_stage2_ema_curve(y_raw, block_size=400, alpha=0.35)
            print(
                f"[Fig26][{name}] raw_episode_len={len(y_raw)}, "
                f"stage_count={n_stage}, plotted_points={len(y_plot)}"
            )
            if y_plot.size == 0:
                continue
            plt.plot(
                x_plot,
                y_plot,
                label=name,
                color=self.color_map.get(name, None),
                linewidth=2.5,
            )
            fig26_has_curve = True

        if not fig26_has_curve:
            print("[Warn][Fig26] no valid curves to plot.")
        plt.title("Average Reward Convergence")
        plt.xlabel("Training Stage (each stage = 400 episodes)")
        plt.ylabel("Reward")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "Fig26_Reward_Convergence_RefStyle.png"), dpi=300)
        plt.close()

        # Fig27: HFR
        print("[Fig27] Ref-style convergence starts from stage 2 for statistical stability.")
        plt.figure(figsize=(10, 5))
        fig27_has_curve = False
        for name in self.roster:
            key = f"{name}_hfr"
            if key not in evidence.files:
                print(f"[Warn][Fig27][{name}] missing key '{key}'. available keys: {files}")
                continue
            y_raw = safe_seed_mean(evidence[key])
            x_plot, y_plot, n_stage = make_stage2_ema_curve(y_raw, block_size=400, alpha=0.35)
            print(
                f"[Fig27][{name}] raw_episode_len={len(y_raw)}, "
                f"stage_count={n_stage}, plotted_points={len(y_plot)}"
            )
            if y_plot.size == 0:
                continue
            plt.plot(
                x_plot,
                y_plot,
                label=name,
                color=self.color_map.get(name, None),
                linewidth=2.5,
            )
            fig27_has_curve = True

        if not fig27_has_curve:
            print("[Warn][Fig27] no valid curves to plot.")
        plt.title("HFR Convergence")
        plt.xlabel("Training Stage (each stage = 400 episodes)")
        plt.ylabel("Failure Rate")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "Fig27_HFR_Convergence_RefStyle.png"), dpi=300)
        plt.close()

        # Fig28: PPR
        print("[Fig28] Ref-style convergence starts from stage 2 for statistical stability.")
        plt.figure(figsize=(10, 5))
        fig28_has_curve = False
        for name in self.roster:
            key = f"{name}_ppr"
            if key not in evidence.files:
                print(f"[Warn][Fig28][{name}] missing key '{key}'. available keys: {files}")
                continue
            y_raw = safe_seed_mean(evidence[key])
            x_plot, y_plot, n_stage = make_stage2_ema_curve(y_raw, block_size=400, alpha=0.35)
            print(
                f"[Fig28][{name}] raw_episode_len={len(y_raw)}, "
                f"stage_count={n_stage}, plotted_points={len(y_plot)}"
            )
            if y_plot.size == 0:
                continue
            plt.plot(
                x_plot,
                y_plot,
                label=name,
                color=self.color_map.get(name, None),
                linewidth=2.5,
            )
            fig28_has_curve = True

        if not fig28_has_curve:
            print("[Warn][Fig28] no valid curves to plot.")
        plt.title("PPR Convergence")
        plt.xlabel("Training Stage (each stage = 400 episodes)")
        plt.ylabel("Ping-pong Rate")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "Fig28_PPR_Convergence_RefStyle.png"), dpi=300)
        plt.close()

        # Fig29: EHR
        print("[Fig29] Ref-style convergence starts from stage 2 for statistical stability.")
        plt.figure(figsize=(10, 5))
        fig29_has_curve = False
        for name in self.roster:
            ehr_key = f"{name}_ehr"
            if ehr_key in evidence.files:
                ehr_raw = safe_seed_mean(evidence[ehr_key])
                source = ehr_key
            else:
                hfr_key = f"{name}_hfr"
                ppr_key = f"{name}_ppr"
                if (hfr_key not in evidence.files) or (ppr_key not in evidence.files):
                    print(f"[Warn][Fig29][{name}] missing ehr/hfr/ppr keys. available keys: {files}")
                    continue
                hfr_raw = safe_seed_mean(evidence[hfr_key])
                ppr_raw = safe_seed_mean(evidence[ppr_key])
                min_len = min(len(hfr_raw), len(ppr_raw))
                if min_len == 0:
                    print(f"[Warn][Fig29][{name}] empty hfr/ppr sequence for fallback ehr.")
                    continue
                ehr_raw = 1.0 - hfr_raw[:min_len] - ppr_raw[:min_len]
                source = f"1-{hfr_key}-{ppr_key}"

            x_plot, y_plot, n_stage = make_stage2_ema_curve(ehr_raw, block_size=400, alpha=0.35)
            print(
                f"[Fig29][{name}] raw_episode_len={len(ehr_raw)}, "
                f"stage_count={n_stage}, plotted_points={len(y_plot)}, source={source}"
            )
            if y_plot.size == 0:
                continue
            plt.plot(
                x_plot,
                y_plot,
                label=name,
                color=self.color_map.get(name, None),
                linewidth=2.5,
            )
            fig29_has_curve = True

        if not fig29_has_curve:
            print("[Warn][Fig29] no valid curves to plot.")
        plt.title("EHR Convergence")
        plt.xlabel("Training Stage (each stage = 400 episodes)")
        plt.ylabel("Effective Handover Ratio")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "Fig29_EHR_Convergence_RefStyle.png"), dpi=300)
        plt.close()

        # Fig30: Reward Gain over Initial Stage (start from stage 1)
        print("[Fig30] Reward gain is computed from stage 1 baseline (R0).")
        plt.figure(figsize=(10, 5))
        fig30_has_curve = False
        for name in self.roster:
            reward_key = None
            for k in [f"{name}_reward", f"{name}_r", f"{name}_overall_reward", f"{name}_rewards"]:
                if k in evidence.files:
                    reward_key = k
                    break
            if reward_key is None:
                print(f"[Warn][Fig30][{name}] reward key missing. available keys: {files}")
                continue

            y_raw = safe_seed_mean(evidence[reward_key])
            y_stage = block_mean(y_raw, block_size=400)
            if y_stage.size == 0:
                print(f"[Warn][Fig30][{name}] no complete 400-episode stage.")
                continue

            y_stage_ema = ema_smooth(y_stage, alpha=0.35)
            r0 = float(y_stage_ema[0])
            y_gain = y_stage_ema - r0
            x_plot = np.arange(0, y_gain.size, dtype=int)

            print(
                f"[Fig30][{name}] R0={r0:.6f}, "
                f"final_reward_gain={float(y_gain[-1]):.6f}, "
                f"raw_episode_len={len(y_raw)}, stage_count={len(y_stage)}, plotted_points={len(y_gain)}"
            )

            plt.plot(
                x_plot,
                y_gain,
                label=name,
                color=self.color_map.get(name, None),
                linewidth=2.5,
            )
            fig30_has_curve = True

        if not fig30_has_curve:
            print("[Warn][Fig30] no valid curves to plot.")
        plt.title("Reward Gain over Initial Stage")
        plt.xlabel("Training Stage (each stage = 400 episodes)")
        plt.ylabel("Reward Gain")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "Fig30_Reward_Gain.png"), dpi=300)
        plt.close()


def plot_fig31_reward_skip_first(data, models, folder, highway_baseline, urban_baseline, smooth=200):
    def rolling_mean(x, window=200):
        import pandas as pd
        return pd.Series(x, dtype=float).rolling(window, min_periods=1).mean().values

    def smooth_series(arr, window=200):
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0:
            return arr
        window = int(max(1, min(window, len(arr))))
        return rolling_mean(arr, window=window)

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

    def scenario_split_from_mask(values_2d, scen_2d, target_scene):
        values_2d = np.asarray(values_2d, dtype=float)
        scen_2d = np.asarray(scen_2d)
        seqs = []
        for s in range(values_2d.shape[0]):
            mask = (scen_2d[s] == target_scene)
            seqs.append(values_2d[s][mask])
        return np.array(seqs, dtype=object)

    def get_scene_reward_series(data_, model_name, scene):
        prefix = "h" if scene == "Highway" else "u"
        key_r = f"{model_name}_{prefix}_r"
        if key_r in data_.files:
            return data_[key_r]
        r2d = np.asarray(data_[f"{model_name}_r"], dtype=float)
        scen2d = np.asarray(data_[f"{model_name}_scen"])
        return scenario_split_from_mask(r2d, scen2d, scene)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    model_color = {
        "DQN (MLP)": "tab:blue",
        "DQN (LSTM)": "tab:orange",
    }

    scen_cfg = [
        ("Highway", highway_baseline, axes[0], "Highway Reward Gain"),
        ("Urban", urban_baseline, axes[1], "Urban Reward Gain"),
    ]

    for scene, baseline, ax, title in scen_cfg:
        for m in models:
            reward_series = get_scene_reward_series(data, m, scene)
            mean_curve, _ = pad_nan_and_nanmean_std(reward_series)
            before_len = int(len(mean_curve))
            if len(mean_curve) > 1:
                mean_curve = mean_curve[1:]
            after_len = int(len(mean_curve))
            print(f"[Fig31][{scene}][{m}] len_before={before_len}, len_after={after_len}")
            if after_len == 0:
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
            label="Zero-gain line (equal to fixed baseline)",
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
    out_path = f"{outdir}/Fig31_Reward.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Fig31] saved file path: {out_path}")


def cumulative_mean_curve(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    return np.cumsum(arr) / (np.arange(arr.size) + 1.0)


def _block_mean(x, block_size=400):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return np.array([], dtype=float)
    n_full = int(x.size // block_size)
    if n_full <= 0:
        return np.array([], dtype=float)
    trimmed = x[: n_full * block_size].reshape(n_full, block_size)
    return np.nanmean(trimmed, axis=1)


def _ema_smooth(data, alpha=0.35):
    data = np.asarray(data, dtype=float).reshape(-1)
    if data.size == 0:
        return np.array([], dtype=float)
    out = np.empty_like(data, dtype=float)
    out[0] = data[0]
    for i in range(1, data.size):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
    return out


def _make_stage2_ema_curve(y_raw, block_size=400, alpha=0.35):
    y_stage = _block_mean(y_raw, block_size=block_size)
    n_stage = int(y_stage.size)
    if n_stage <= 1:
        return np.array([], dtype=float), np.array([], dtype=float), n_stage
    y_stage = y_stage[1:]  # stage 2 onwards
    y_plot = _ema_smooth(y_stage, alpha=alpha)
    x_plot = np.arange(2, 2 + y_plot.size, dtype=int)
    return x_plot, y_plot, n_stage


def _nanmean_curve_from_series(series):
    arr = np.asarray(series)

    if arr.size == 0:
        return np.array([], dtype=float)

    if arr.dtype != object:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            return arr
        if arr.ndim >= 2:
            return np.nanmean(arr, axis=0)
        return np.array([], dtype=float)

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
        return np.array([], dtype=float)

    max_len = max(len(x) for x in seqs)
    padded = np.full((len(seqs), max_len), np.nan, dtype=float)
    for i, x in enumerate(seqs):
        padded[i, :len(x)] = x
    return np.nanmean(padded, axis=0)


def _scenario_split_from_mask(values_2d, scen_2d, target_scene):
    values_2d = np.asarray(values_2d, dtype=float)
    scen_2d = np.asarray(scen_2d)

    if values_2d.ndim != 2 or scen_2d.ndim != 2:
        return np.array([], dtype=object)

    seed_count = min(values_2d.shape[0], scen_2d.shape[0])
    seqs = []
    for s in range(seed_count):
        mask = scen_2d[s] == target_scene
        seqs.append(values_2d[s][mask])
    return np.array(seqs, dtype=object)


def _get_scene_metric_series(data, model_name, scene, metric):
    scene_lower = scene.lower()
    prefix = "h" if scene == "Highway" else "u"

    direct_candidates = [
        f"{model_name}_{scene_lower}_{metric}",
        f"{model_name}_{prefix}_{metric}",
    ]
    for key in direct_candidates:
        if key in data.files:
            return data[key]

    overall_key = f"{model_name}_{metric}"
    scen_key = f"{model_name}_scen"
    if (overall_key not in data.files) or (scen_key not in data.files):
        return np.array([], dtype=object)

    values_2d = np.asarray(data[overall_key], dtype=float)
    scen_2d = np.asarray(data[scen_key])
    return _scenario_split_from_mask(values_2d, scen_2d, scene)


def _get_scene_metric_cumulative_curve(data, model_name, scene, metric):
    series = _get_scene_metric_series(data, model_name, scene, metric)
    mean_curve = _nanmean_curve_from_series(series)
    if mean_curve.size == 0:
        return mean_curve
    return cumulative_mean_curve(mean_curve)


def plot_fig32_hfr_cumulative_by_scene(data, models, folder):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    model_color = {
        "DQN (MLP)": "tab:blue",
        "DQN (LSTM)": "tab:orange",
    }

    for scene, ax, title in [
        ("Highway", axes[0], "Highway HFR Convergence"),
        ("Urban", axes[1], "Urban HFR Convergence"),
    ]:
        for m in models:
            series = _get_scene_metric_series(data, m, scene, "hfr")
            mean_curve = _nanmean_curve_from_series(series)
            x_plot, y_plot, n_stage = _make_stage2_ema_curve(mean_curve, block_size=400, alpha=0.35)
            if y_plot.size == 0:
                print(f"[Warn][Fig32][{scene}][{m}] empty curve, skip.")
                continue
            print(
                f"[Fig32][{scene}][{m}] raw_episode_len={len(mean_curve)}, "
                f"stage_count={n_stage}, plotted_points={len(y_plot)}"
            )
            ax.plot(x_plot, y_plot, linewidth=2.5, color=model_color.get(m, None), label=m)

        ax.set_title(title)
        ax.set_ylabel("Failure Rate")
        ax.set_xlabel("Training Stage (each stage = 400 episodes)")
        ax.grid(True, alpha=0.3)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper left")

    outdir = f"Result/{folder}/Plots"
    os.makedirs(outdir, exist_ok=True)
    out_path = f"{outdir}/Fig32_HFR_Cumulative_ByScene.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Fig32] saved file path: {out_path}")


def plot_fig33_ppr_cumulative_by_scene(data, models, folder):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    model_color = {
        "DQN (MLP)": "tab:blue",
        "DQN (LSTM)": "tab:orange",
    }

    for scene, ax, title in [
        ("Highway", axes[0], "Highway PPR Convergence"),
        ("Urban", axes[1], "Urban PPR Convergence"),
    ]:
        for m in models:
            series = _get_scene_metric_series(data, m, scene, "ppr")
            mean_curve = _nanmean_curve_from_series(series)
            x_plot, y_plot, n_stage = _make_stage2_ema_curve(mean_curve, block_size=400, alpha=0.35)
            if y_plot.size == 0:
                print(f"[Warn][Fig33][{scene}][{m}] empty curve, skip.")
                continue
            print(
                f"[Fig33][{scene}][{m}] raw_episode_len={len(mean_curve)}, "
                f"stage_count={n_stage}, plotted_points={len(y_plot)}"
            )
            ax.plot(x_plot, y_plot, linewidth=2.5, color=model_color.get(m, None), label=m)

        ax.set_title(title)
        ax.set_ylabel("Ping-pong Rate")
        ax.set_xlabel("Training Stage (each stage = 400 episodes)")
        ax.grid(True, alpha=0.3)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper left")

    outdir = f"Result/{folder}/Plots"
    os.makedirs(outdir, exist_ok=True)
    out_path = f"{outdir}/Fig33_PPR_Cumulative_ByScene.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Fig33] saved file path: {out_path}")


def find_latest_tournament_folder(result_base="Result"):
    if not os.path.isdir(result_base):
        return None
    folders = [
        f for f in os.listdir(result_base)
        if os.path.isdir(os.path.join(result_base, f)) and f.startswith("Tournament_")
    ]
    if not folders:
        return None
    return sorted(folders)[-1]


def main():
    result_base = "Result"
    latest_folder = find_latest_tournament_folder(result_base=result_base)
    if latest_folder is None:
        print("No Tournament folder found.")
        return

    folder_path = os.path.join(result_base, latest_folder)
    npz_path = os.path.join(folder_path, "raw_evidence.npz")
    if not os.path.exists(npz_path):
        print("Missing raw_evidence.npz.")
        return

    evidence = np.load(npz_path, allow_pickle=True)
    save_dir = os.path.join(folder_path, "Plots")
    roster = ["DQN (MLP)", "DQN (LSTM)"]

    # Existing flow (if any) should run before this call; this call appends Fig26-29 generation.
    plotter = RefStyleConvergencePlotter(save_dir=save_dir, roster=roster)
    plotter.plot_refstyle_convergence_with_stage_ema(evidence)

    print("Fig26_Reward_Convergence_RefStyle.png")
    print("Fig27_HFR_Convergence_RefStyle.png")
    print("Fig28_PPR_Convergence_RefStyle.png")
    print("Fig29_EHR_Convergence_RefStyle.png")

    highway_baseline, urban_baseline = None, None
    summary_path = "Result/Stage1_Baseline/baseline_summary.csv"
    metrics_path = "Result/Stage1_Baseline/baseline_metrics.csv"
    try:
        import pandas as pd
        if os.path.exists(summary_path) and os.path.exists(metrics_path):
            summary_df = pd.read_csv(summary_path)
            metrics_df = pd.read_csv(metrics_path)
            if (not summary_df.empty) and ("baseline_threshold_dBm" in summary_df.columns) and ("threshold_dBm" in metrics_df.columns):
                threshold = pd.to_numeric(summary_df.iloc[-1]["baseline_threshold_dBm"], errors="coerce")
                thresholds = pd.to_numeric(metrics_df["threshold_dBm"], errors="coerce").to_numpy(dtype=float)
                valid_idx = np.where(np.isfinite(thresholds))[0]
                if np.isfinite(threshold) and valid_idx.size > 0:
                    close_idx = valid_idx[np.isclose(thresholds[valid_idx], float(threshold), atol=1e-8)]
                    if close_idx.size > 0:
                        idx = int(close_idx[0])
                    else:
                        idx = int(valid_idx[np.argmin(np.abs(thresholds[valid_idx] - float(threshold)))])
                    highway_baseline = pd.to_numeric(metrics_df.iloc[idx].get("highway_score", np.nan), errors="coerce")
                    urban_baseline = pd.to_numeric(metrics_df.iloc[idx].get("urban_score", np.nan), errors="coerce")
    except Exception:
        highway_baseline, urban_baseline = None, None

    if (highway_baseline is None) or (urban_baseline is None) or (not np.isfinite(highway_baseline)) or (not np.isfinite(urban_baseline)):
        print("[Warn][Fig31] Missing Stage1 baseline reward; skip Fig31_Reward.png")
    else:
        plot_fig31_reward_skip_first(
            data=evidence,
            models=roster,
            folder=latest_folder,
            highway_baseline=float(highway_baseline),
            urban_baseline=float(urban_baseline),
            smooth=200,
        )

    plot_fig32_hfr_cumulative_by_scene(evidence, roster, latest_folder)
    plot_fig33_ppr_cumulative_by_scene(evidence, roster, latest_folder)


if __name__ == "__main__":
    main()
