import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def find_latest_tournament_folder():
    prefixes = ("Tournament_10k_", "Tournament_8000_", "Tournament_Parallel")
    if not os.path.isdir("Result"):
        return None
    folders = [f for f in os.listdir("Result")
               if os.path.isdir(f"Result/{f}") and f.startswith(prefixes)]
    if not folders:
        return None
    return sorted(folders)[-1]


def load_stage1_baseline_files():
    metrics_path = "Result/Stage1_Baseline/baseline_metrics.csv"
    summary_path = "Result/Stage1_Baseline/baseline_summary.csv"
    metrics_df = None
    summary_row = None

    if os.path.exists(metrics_path):
        tmp = pd.read_csv(metrics_path)
        required = {"threshold_dBm", "highway_score", "urban_score"}
        if required.issubset(set(tmp.columns)):
            metrics_df = tmp
        else:
            print(f"[Warn] baseline_metrics.csv missing columns: {sorted(required - set(tmp.columns))}")

    if os.path.exists(summary_path):
        tmp = pd.read_csv(summary_path)
        required = {"best_highway_score", "best_urban_score", "best_highway_threshold", "best_urban_threshold"}
        if len(tmp) > 0 and required.issubset(set(tmp.columns)):
            summary_row = tmp.iloc[0]
        else:
            print(f"[Warn] baseline_summary.csv missing columns: {sorted(required - set(tmp.columns))}")

    return metrics_df, summary_row, metrics_path, summary_path


def get_baseline_kpi_from_summary(summary_row):
    if summary_row is None:
        return None
    for c_hfr, c_ppr, c_ehr in [
        ("baseline_hfr", "baseline_ppr", "baseline_ehr"),
        ("HFR", "PPR", "EHR"),
    ]:
        if all(c in summary_row.index for c in [c_hfr, c_ppr, c_ehr]):
            return float(summary_row[c_hfr]), float(summary_row[c_ppr]), float(summary_row[c_ehr])
    return None


def to_percent_scalar(v):
    v = float(v)
    return v * 100.0 if abs(v) <= 1.5 else v


def get_model_value(df, model_name, column):
    rows = df.loc[df["Model"] == model_name, column]
    if len(rows) == 0:
        return np.nan
    return float(rows.iloc[0])


def plot_three_bar(title, ylabel, baseline_val, mlp_val, lstm_val, filename, higher_is_better=True):
    fig, ax = plt.subplots(figsize=(7, 5))
    models = ["Fixed\nBaseline", "DQN\n(MLP)", "DQN\n(LSTM)"]
    values = [baseline_val, mlp_val, lstm_val]
    colors = ['#888888', '#1f77b4', '#ff7f0e']
    bars = ax.bar(models, values, color=colors, width=0.5, edgecolor='black', linewidth=0.8)

    finite_vals = [v for v in values if np.isfinite(v)]
    top = max(finite_vals) if finite_vals else 1.0
    text_off = top * 0.01 if top > 0 else 0.01

    for bar, val in zip(bars, values):
        if np.isfinite(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + text_off,
                f"{val:.2f}%",
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )

    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, max(top * 1.2, 1.0))
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    folder = find_latest_tournament_folder()
    if folder is None:
        print("No tournament folders found (expected prefixes: Tournament_10k_, Tournament_8000_, Tournament_Parallel).")
        return

    plots_dir = f"Result/{folder}/Plots"
    os.makedirs(plots_dir, exist_ok=True)

    df_path = f"Result/{folder}/final_leaderboard.csv"
    if not os.path.exists(df_path):
        print(f"Missing final_leaderboard.csv in {folder}")
        return
    df = pd.read_csv(df_path)
    required_ai_cols = {"Model", "Highway Score", "Urban Score"}
    if not required_ai_cols.issubset(set(df.columns)):
        print(f"Missing required columns in final_leaderboard.csv: {sorted(required_ai_cols - set(df.columns))}")
        return

    metrics_df, summary_row, metrics_path, summary_path = load_stage1_baseline_files()

    fallback_hfr, fallback_ppr, fallback_ehr = 0.15, 0.12, 0.725
    baseline_kpi = get_baseline_kpi_from_summary(summary_row)
    if baseline_kpi is not None:
        baseline_hfr, baseline_ppr, baseline_ehr = baseline_kpi
        print(f"Loaded KPI baseline from: {summary_path}")
    else:
        baseline_hfr, baseline_ppr, baseline_ehr = fallback_hfr, fallback_ppr, fallback_ehr
        if summary_row is not None:
            print(f"[Warn] {summary_path} found but KPI columns missing; fallback to hardcoded KPI baseline.")
        else:
            print("[Warn] Stage1 baseline summary not found; fallback to hardcoded KPI baseline.")

    mlp_ppr = get_model_value(df, "DQN (MLP)", "PPR")
    lstm_ppr = get_model_value(df, "DQN (LSTM)", "PPR")
    mlp_hfr = get_model_value(df, "DQN (MLP)", "HFR")
    lstm_hfr = get_model_value(df, "DQN (LSTM)", "HFR")
    mlp_ehr = get_model_value(df, "DQN (MLP)", "EHR")
    lstm_ehr = get_model_value(df, "DQN (LSTM)", "EHR")

    plot_three_bar(
        title="Final PPR Comparison",
        ylabel="Rate (%)",
        baseline_val=to_percent_scalar(baseline_ppr),
        mlp_val=to_percent_scalar(mlp_ppr),
        lstm_val=to_percent_scalar(lstm_ppr),
        filename=f"{plots_dir}/Fig13_PPR_Bar.png",
        higher_is_better=False,
    )
    plot_three_bar(
        title="Final HFR Comparison",
        ylabel="Rate (%)",
        baseline_val=to_percent_scalar(baseline_hfr),
        mlp_val=to_percent_scalar(mlp_hfr),
        lstm_val=to_percent_scalar(lstm_hfr),
        filename=f"{plots_dir}/Fig14_HFR_Bar.png",
        higher_is_better=False,
    )
    plot_three_bar(
        title="Final EHR Comparison",
        ylabel="Effectiveness (%)",
        baseline_val=to_percent_scalar(baseline_ehr),
        mlp_val=to_percent_scalar(mlp_ehr),
        lstm_val=to_percent_scalar(lstm_ehr),
        filename=f"{plots_dir}/Fig15_EHR_Bar.png",
        higher_is_better=True,
    )

    plt.figure(figsize=(9, 7))

    x_pool, y_pool = [], []
    if metrics_df is not None:
        bx = np.asarray(metrics_df["highway_score"], dtype=float)
        by = np.asarray(metrics_df["urban_score"], dtype=float)
        if bx.size > 0 and by.size > 0:
            plt.plot(bx, by, 'b--', alpha=0.35, label='Static baseline envelope')
            plt.scatter(bx, by, s=45, alpha=0.35, color='tab:blue', edgecolors='none')
            x_pool.append(bx)
            y_pool.append(by)
            print(f"Loaded static baseline points from: {metrics_path}")

    ai_points = df.loc[:, ["Model", "Highway Score", "Urban Score"]].copy()
    ai_points["Highway Score"] = pd.to_numeric(ai_points["Highway Score"], errors="coerce")
    ai_points["Urban Score"] = pd.to_numeric(ai_points["Urban Score"], errors="coerce")

    for _, row in ai_points.iterrows():
        plt.scatter(
            row['Highway Score'],
            row['Urban Score'],
            s=600,
            marker='*',
            edgecolors='black',
            label=f"AI: {row['Model']}"
        )
        plt.annotate(
            row['Model'],
            (row['Highway Score'], row['Urban Score']),
            xytext=(10, 10),
            textcoords='offset points',
            weight='bold'
        )

    ai_x = np.asarray(ai_points['Highway Score'].values, dtype=float)
    ai_y = np.asarray(ai_points['Urban Score'].values, dtype=float)
    x_pool.append(ai_x)
    y_pool.append(ai_y)

    x_list = [arr[np.isfinite(arr)] for arr in x_pool if arr.size > 0]
    y_list = [arr[np.isfinite(arr)] for arr in y_pool if arr.size > 0]
    x_all = np.concatenate(x_list) if x_list else np.array([])
    y_all = np.concatenate(y_list) if y_list else np.array([])
    if x_all.size > 0 and y_all.size > 0:
        pad_x = (x_all.max() - x_all.min()) * 0.15 + 1e-6
        pad_y = (y_all.max() - y_all.min()) * 0.15 + 1e-6
        plt.xlim(x_all.min() - pad_x, x_all.max() + pad_x)
        plt.ylim(y_all.min() - pad_y, y_all.max() + pad_y)

    plt.xlabel("Highway Score")
    plt.ylabel("Urban Score")
    plt.title("Pareto Envelope and AI Adaptation")
    plt.legend(loc='lower left')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/05_Ultimate_Leap.png", dpi=300)
    plt.close()

    print("ALL EXPERIMENTS COMPLETE!")
    print("Fig 13-15 and Ultimate Leap Generated.")


if __name__ == "__main__":
    main()
