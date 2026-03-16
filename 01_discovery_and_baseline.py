import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vehicle_env import VehicleEnv
import config
import os

def run_ep(env, idx, scen):
    env.reset(force_scenario=scen)
    score, done = 0, False
    while not done:
        _, r, done, _ = env.step(idx)
        score += r
    return score

def run_ep_with_kpi(env, idx, scen):
    env.reset(force_scenario=scen)
    score, done = 0.0, False
    ep_ho_fail, ep_pp, ep_ho = 0, 0, 0
    while not done:
        _, r, done, info = env.step(idx)
        score += float(r)
        ep_ho_fail += int(info.get('is_rlf', 0))
        ep_pp += int(info.get('is_pingpong', 0))
        ep_ho += int(info.get('ho_attempted', 0))  # 与02一致
    hfr = ep_ho_fail / max(ep_ho, 1)
    ppr = ep_pp / (ep_ho + 1e-6)
    ehr = float(np.clip(1.0 - hfr - ppr, 0.0, 1.0))
    return score, hfr, ppr, ehr

def main():
    import random
    np.random.seed(250)
    random.seed(250)
    print("Stage 0 & 1: Protocol Sanity & Pareto Baseline (Optimized)")
    env = VehicleEnv()
    os.makedirs("Result", exist_ok=True)
    thresholds = config.ACTION_THRESHOLDS
    labels = [f"{t}dBm" for t in thresholds]

    # --- 1. 01a_Protocol_Sanity (协议逻辑验证：阈值下限，增加触发标注) ---
    print("  Running Stage 01a...")
    env.reset(force_scenario='Urban')
    test_thresh_val = -100 # 阈值下限，确保信号能够触发条件判断
    test_idx = 3 # 对应 -105dBm
    
    rsrp_h, ttt_h = [], []
    trig_step = None
    for t in range(150):
        _, _, _, info = env.step(test_idx)
        rsrp_h.append(info['rsrp'])
        ttt_h.append(info['ttt'])
        if info['ttt'] >= config.TTT_STEPS and trig_step is None:
            trig_step = t

    plt.figure(figsize=(10, 5))
    plt.plot(rsrp_h, label='Measured SL-RSRP', color='tab:blue', linewidth=1.5)
    plt.axhline(y=test_thresh_val, color='red', linestyle='--', label=f'IE: sl-RSRP-Thresh ({test_thresh_val}dBm)')
    
    if trig_step is not None:
        # 新增：标注 TTT 影响窗口和触发点，图示更直观
        plt.scatter(trig_step, rsrp_h[trig_step], color='red', s=100, zorder=5, label='3GPP Trigger Point')
        plt.axvspan(trig_step - config.TTT_STEPS, trig_step, color='red', alpha=0.1, label='TTT Window')
        
    plt.title("TS 38.331 Protocol Logic Verification (Standard Compliance)", fontsize=12)
    plt.xlabel("Time Step (100ms)"); plt.ylabel("RSRP (dBm)")
    plt.legend(loc='upper right', fontsize=9); plt.grid(True, alpha=0.2)
    plt.savefig("Result/01a_Protocol_Sanity.png", dpi=300)

    # --- 2. 01b_Crossover_Proof (交叉验证：增加最优阈值文字标注) ---
    print("  Running Stage 01b...")
    h_scores, u_scores = [], []
    for i in range(len(thresholds)):
        h_scores.append(np.mean([run_ep(env, i, 'Highway') for _ in range(100)]))
        u_scores.append(np.mean([run_ep(env, i, 'Urban') for _ in range(100)]))

    # KPI per static threshold (episode-level aggregation, consistent with Stage 02)
    baseline_hfr, baseline_ppr, baseline_ehr = [], [], []
    highway_hfr, urban_hfr = [], []
    highway_ppr, urban_ppr = [], []
    highway_ehr, urban_ehr = [], []
    for i in range(len(thresholds)):
        h_kpi = [run_ep_with_kpi(env, i, 'Highway') for _ in range(100)]
        u_kpi = [run_ep_with_kpi(env, i, 'Urban') for _ in range(100)]
        h_hfr = [x[1] for x in h_kpi]
        h_ppr = [x[2] for x in h_kpi]
        h_ehr = [x[3] for x in h_kpi]
        u_hfr = [x[1] for x in u_kpi]
        u_ppr = [x[2] for x in u_kpi]
        u_ehr = [x[3] for x in u_kpi]
        # scene KPI
        highway_hfr.append(float(np.mean(h_hfr)))
        urban_hfr.append(float(np.mean(u_hfr)))

        highway_ppr.append(float(np.mean(h_ppr)))
        urban_ppr.append(float(np.mean(u_ppr)))

        highway_ehr.append(float(np.mean(h_ehr)))
        urban_ehr.append(float(np.mean(u_ehr)))

        # overall KPI (保持原逻辑)
        baseline_hfr.append(float(np.mean(h_hfr + u_hfr)))
        baseline_ppr.append(float(np.mean(h_ppr + u_ppr)))
        baseline_ehr.append(float(np.mean(h_ehr + u_ehr)))

    plt.figure(figsize=(11, 6))
    plt.plot(thresholds, h_scores, 'o-', label='Highway (LOS/High-Speed)', linewidth=2, markersize=8)
    plt.plot(thresholds, u_scores, 's-', label='Urban (NLOS/Blockage)', linewidth=2, markersize=8)
    
    # 新增最优点标注
    h_best_idx = np.argmax(h_scores)
    u_best_idx = np.argmax(u_scores)
    plt.text(thresholds[h_best_idx], h_scores[h_best_idx]+200, f"Best: {thresholds[h_best_idx]}dBm", 
             ha='center', color='blue', weight='bold')
    plt.text(thresholds[u_best_idx], u_scores[u_best_idx]+200, f"Best: {thresholds[u_best_idx]}dBm", 
             ha='center', color='orange', weight='bold')

    plt.title("Performance Crossover Proof", fontsize=13)
    plt.xlabel("Static Threshold IE (dBm)"); plt.ylabel("Cumulative Reward")
    plt.legend(); plt.grid(True, alpha=0.3); plt.savefig("Result/01b_Crossover_Proof.png", dpi=300)

    # --- 3. 01c_Pareto_Baseline (帕累托基线：静态门限包络) ---
    print("  Running Stage 01c...")
    plt.figure(figsize=(10, 8))
    plt.plot(h_scores, u_scores, 'b--o', alpha=0.4, label='Static 3GPP Limit Envelope')
    for i, txt in enumerate(labels):
        plt.annotate(txt, (h_scores[i], u_scores[i]), xytext=(5, -15), textcoords='offset points', color='blue', fontsize=9)
    
    h_max, u_max = max(h_scores), max(u_scores)
   
    
    plt.xlabel("Highway Performance (Throughput)"); plt.ylabel("Urban Performance (Reliability)")
    plt.title("The Physical Trade-off Barrier", fontsize=13, fontweight='bold')
    
    # 调整坐标显示范围
    plt.xlim(min(h_scores)*0.98, h_max * 1.12)
    plt.ylim(min(u_scores)*0.95, u_max * 1.15)
    
    plt.legend(loc='lower left'); plt.grid(True, linestyle=':', alpha=0.5)
    plt.savefig("Result/01c_Pareto_Baseline.png", dpi=300)

    # 导出关键数据供 05 脚本使用
    np.savez("Result/fixed_baseline_data.npz", h_fixed=h_scores, u_fixed=u_scores, thresholds=thresholds, h_max=h_max, u_max=u_max)

    # === Stage 01c: Standardized Baseline Export (for Stage-2/4) ===
    stage1_baseline_dir = "Result/Stage1_Baseline"
    os.makedirs(stage1_baseline_dir, exist_ok=True)

    baseline_metrics_df = pd.DataFrame({
        "threshold_dBm": thresholds,
        "highway_score": h_scores,
        "urban_score": u_scores,

        "highway_hfr": highway_hfr,
        "urban_hfr": urban_hfr,

        "highway_ppr": highway_ppr,
        "urban_ppr": urban_ppr,

        "highway_ehr": highway_ehr,
        "urban_ehr": urban_ehr,

        "HFR": baseline_hfr,
        "PPR": baseline_ppr,
        "EHR": baseline_ehr,
    })
    baseline_metrics_df.to_csv(f"{stage1_baseline_dir}/baseline_metrics.csv", index=False)

    # Pareto sweet-spot threshold (balanced trade-off on normalized Highway/Urban scores)
    h_norm = np.array(h_scores, dtype=float) / max(h_max, 1e-6)
    u_norm = np.array(u_scores, dtype=float) / max(u_max, 1e-6)
    sweet_spot_idx = int(np.argmax(np.minimum(h_norm, u_norm)))

    baseline_summary_df = pd.DataFrame([{
        "best_highway_score": h_max,
        "best_urban_score": u_max,
        "best_highway_threshold": thresholds[h_best_idx],
        "best_urban_threshold": thresholds[u_best_idx],
        "baseline_threshold_dBm": thresholds[sweet_spot_idx],
        "baseline_hfr": float(baseline_hfr[sweet_spot_idx]),
        "baseline_ppr": float(baseline_ppr[sweet_spot_idx]),
        "baseline_ehr": float(baseline_ehr[sweet_spot_idx]),
    }])
    baseline_summary_df.to_csv(f"{stage1_baseline_dir}/baseline_summary.csv", index=False)
    print("? All Baseline stages finished successfully.")

if __name__ == "__main__": main()
