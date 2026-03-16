import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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


def block_mean(x, block_size):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return np.array([], dtype=float)
    n_full = int(x.size // block_size)
    if n_full <= 0:
        return np.array([], dtype=float)
    trimmed = x[:n_full * block_size].reshape(n_full, block_size)
    return np.nanmean(trimmed, axis=1)


def cumulative_mean(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return np.array([], dtype=float)
    return np.cumsum(x) / (np.arange(x.size) + 1)


def rolling_mean(x, window=2):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return np.array([], dtype=float)
    return pd.Series(x).rolling(window=window, min_periods=1).mean().values


def prepare_stage_curve(y_raw, start_stage=1, block_size=400, smooth_window=2):
    y_stage = block_mean(y_raw, block_size=block_size)
    y_cum = cumulative_mean(y_stage)
    y_plot = rolling_mean(y_cum, window=smooth_window)

    start_idx = max(0, int(start_stage) - 1)
    if y_plot.size <= start_idx:
        return np.array([], dtype=float), np.array([], dtype=float), int(y_stage.size)

    y_show = y_plot[start_idx:]
    x_show = np.arange(start_stage, start_stage + y_show.size, dtype=int)
    return x_show, y_show, int(y_stage.size)


def find_latest_tournament_folder(result_base="Result"):
    if not os.path.isdir(result_base):
        return None
    folders = [f for f in os.listdir(result_base) if os.path.isdir(os.path.join(result_base, f)) and f.startswith("Tournament_")]
    if not folders:
        return None
    return sorted(folders)[-1]

def main():
    result_base = "Result"
    latest_folder = find_latest_tournament_folder(result_base=result_base)
    if latest_folder is None:
        print("❌ 未找到 Tournament 结果文件夹")
        return

    folder_path = os.path.join(result_base, latest_folder)
    npz_path = os.path.join(folder_path, "raw_evidence.npz")

    if not os.path.exists(npz_path):
        print("❌ 找不到原始数据 npz 文件")
        return

    data = np.load(npz_path, allow_pickle=True)
    models = ["DQN (MLP)", "DQN (LSTM)"]

    save_dir = os.path.join(folder_path, "Plots")
    os.makedirs(save_dir, exist_ok=True)

    # 设置绘图风格
    plt.rcParams['font.family'] = 'serif'
    color_map = {"DQN (MLP)": "red", "DQN (LSTM)": "blue"}
    stage_msg = "plot starts from stage 2 for statistical stability"

    # ==========================================
    # Fig22: Reward Convergence (RefStyle)
    # ==========================================
    plt.figure(figsize=(10, 5))
    fig22_has_curve = False
    for name in models:
        reward_key = None
        for k in [f"{name}_reward", f"{name}_r", f"{name}_overall_reward", f"{name}_rewards"]:
            if k in data.files:
                reward_key = k
                break
        if reward_key is None:
            print(f"[Warn][Fig22][{name}] reward key missing. available keys: {list(data.files)}")
            continue

        y_raw = safe_seed_mean(data[reward_key])
        x_plot, y_plot, n_stage = prepare_stage_curve(y_raw, start_stage=1, block_size=400, smooth_window=2)
        print(f"[Fig22][{name}] raw_episode_len={len(y_raw)}, n_stage={n_stage}, n_points={len(y_plot)}")
        if y_plot.size == 0:
            continue

        plt.plot(x_plot, y_plot, label=name, color=color_map[name], linewidth=2.5)
        fig22_has_curve = True

    if not fig22_has_curve:
        print("[Warn][Fig22] no valid curves to plot.")
    plt.title("Average Reward Convergence", fontsize=12)
    plt.xlabel("Training Stage (each stage = 400 episodes)", fontsize=10)
    plt.ylabel("Reward", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    fig22_name = "Fig22_Reward_Convergence_RefStyle.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fig22_name), dpi=300)
    plt.close()

    # ==========================================
    # Fig23: HFR Convergence (RefStyle)
    # ==========================================
    print(f"[Fig23] {stage_msg}")
    plt.figure(figsize=(10, 5))
    fig23_has_curve = False
    for name in models:
        key = f"{name}_hfr"
        if key not in data.files:
            print(f"[Warn][Fig23][{name}] key missing: {key}. available keys: {list(data.files)}")
            continue
        y_raw = safe_seed_mean(data[key])
        x_plot, y_plot, n_stage = prepare_stage_curve(y_raw, start_stage=2, block_size=400, smooth_window=2)
        print(f"[Fig23][{name}] raw_episode_len={len(y_raw)}, n_stage={n_stage}, n_points={len(y_plot)}")
        if y_plot.size == 0:
            continue

        plt.plot(x_plot, y_plot, label=name, color=color_map[name], linewidth=2.5)
        fig23_has_curve = True

    if not fig23_has_curve:
        print("[Warn][Fig23] no valid curves to plot.")
    plt.title("HFR Convergence", fontsize=12)
    plt.xlabel("Training Stage (each stage = 400 episodes)", fontsize=10)
    plt.ylabel("Failure Rate", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    fig23_name = "Fig23_HFR_Convergence_RefStyle.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fig23_name), dpi=300)
    plt.close()

    # ==========================================
    # Fig24: PPR Convergence (RefStyle)
    # ==========================================
    print(f"[Fig24] {stage_msg}")
    plt.figure(figsize=(10, 5))
    fig24_has_curve = False
    for name in models:
        key = f"{name}_ppr"
        if key not in data.files:
            print(f"[Warn][Fig24][{name}] key missing: {key}. available keys: {list(data.files)}")
            continue
        y_raw = safe_seed_mean(data[key])
        x_plot, y_plot, n_stage = prepare_stage_curve(y_raw, start_stage=2, block_size=400, smooth_window=2)
        print(f"[Fig24][{name}] raw_episode_len={len(y_raw)}, n_stage={n_stage}, n_points={len(y_plot)}")
        if y_plot.size == 0:
            continue

        plt.plot(x_plot, y_plot, label=name, color=color_map[name], linewidth=2.5)
        fig24_has_curve = True

    if not fig24_has_curve:
        print("[Warn][Fig24] no valid curves to plot.")
    plt.title("PPR Convergence", fontsize=12)
    plt.xlabel("Training Stage (each stage = 400 episodes)", fontsize=10)
    plt.ylabel("Ping-pong Rate", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    fig24_name = "Fig24_PPR_Convergence_RefStyle.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fig24_name), dpi=300)
    plt.close()

    # ==========================================
    # Fig25: EHR Convergence (RefStyle)
    # ==========================================
    print(f"[Fig25] {stage_msg}")
    plt.figure(figsize=(10, 5))
    fig25_has_curve = False
    for name in models:
        ehr_key = f"{name}_ehr"
        if ehr_key in data.files:
            ehr_raw = safe_seed_mean(data[ehr_key])
            data_source = ehr_key
        else:
            hfr_key = f"{name}_hfr"
            ppr_key = f"{name}_ppr"
            if (hfr_key not in data.files) or (ppr_key not in data.files):
                print(f"[Warn][Fig25][{name}] ehr/hfr/ppr key missing. available keys: {list(data.files)}")
                continue
            hfr_raw = safe_seed_mean(data[hfr_key])
            ppr_raw = safe_seed_mean(data[ppr_key])
            min_len = min(len(hfr_raw), len(ppr_raw))
            if min_len == 0:
                print(f"[Warn][Fig25][{name}] empty hfr/ppr for fallback ehr.")
                continue
            ehr_raw = 1.0 - hfr_raw[:min_len] - ppr_raw[:min_len]
            data_source = f"1-{hfr_key}-{ppr_key}"

        x_plot, y_plot, n_stage = prepare_stage_curve(ehr_raw, start_stage=2, block_size=400, smooth_window=2)
        print(f"[Fig25][{name}] raw_episode_len={len(ehr_raw)}, n_stage={n_stage}, n_points={len(y_plot)}, source={data_source}")
        if y_plot.size == 0:
            continue

        plt.plot(x_plot, y_plot, label=name, color=color_map[name], linewidth=2.5)
        fig25_has_curve = True

    if not fig25_has_curve:
        print("[Warn][Fig25] no valid curves to plot.")
    plt.title("EHR Convergence", fontsize=12)
    plt.xlabel("Training Stage (each stage = 400 episodes)", fontsize=10)
    plt.ylabel("Effective Handover Ratio", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    fig25_name = "Fig25_EHR_Convergence_RefStyle.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fig25_name), dpi=300)
    plt.close()

    print(f"Successfully generated ref-style plots in: {save_dir}")
    print(fig22_name)
    print(fig23_name)
    print(fig24_name)
    print(fig25_name)

if __name__ == "__main__":
    main()
