import os, time, torch, numpy as np, pandas as pd
from tqdm import tqdm
from datetime import datetime
import config
from vehicle_env import VehicleEnv
from dqn_agent import DQNAgent
from lstm_dqn_agent import LSTMDQNAgent

def train():
    print(f"🚀 Stage 2: {config.MAX_EPISODES} Episode AI Training (Full Data Production, Matrix Format)")
    env = VehicleEnv()
    scenario_block_episodes = 200
    scenario_cycle_episodes = scenario_block_episodes * 2
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = f"Result/Tournament_10k_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    roster = {
        "DQN (MLP)": DQNAgent,
        "DQN (LSTM)": LSTMDQNAgent,
    }
    raw_evidence = {}
    final_report = []

    for name, AgentClass in roster.items():
        print(f"\n--- ⚔️  Training: {name} ---")
        seeds = list(config.SEEDS)

        

        # 规整矩阵容器：每个元素最终都是 (num_seeds, MAX_EPISODES)
        all_r, all_hfr, all_ppr, all_ehr, all_scen = [], [], [], [], []
        all_rlf_steps, all_steps, all_pingpong, all_handover = [], [], [], []
        all_h_r, all_h_hfr, all_h_ppr, all_h_ehr = [], [], [], []
        all_u_r, all_u_hfr, all_u_ppr, all_u_ehr = [], [], [], []
        inference_latencies = []

        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)

            agent = AgentClass()

            # 每个 seed 的 episode 历史（长度=MAX_EPISODES）
            r_hist, hfr_hist, ppr_hist, ehr_hist = [], [], [], []
            rlf_steps_hist, steps_hist, pingpong_hist, handover_hist = [], [], [], []
            scen_hist = []
            h_r_hist, h_hfr_hist, h_ppr_hist, h_ehr_hist = [], [], [], []
            u_r_hist, u_hfr_hist, u_ppr_hist, u_ehr_hist = [], [], [], []

            pbar = tqdm(range(config.MAX_EPISODES), desc=f"{name} Seed {seed}")
            for ep in pbar:
                # 学习率退火（保留你原逻辑）
                if ep == int(config.MAX_EPISODES * 0.7):
                    if hasattr(agent, 'optimizer'):
                        for pg in agent.optimizer.param_groups:
                            pg['lr'] *= 0.1

                # LSTM 记忆清理（保留你原逻辑）
                if hasattr(agent, 'state_sequence'):
                    agent.state_sequence.clear()

                # first 200 episodes Highway, next 200 episodes Urban, then repeat
                curr_scen = "Highway" if (ep % scenario_cycle_episodes) < scenario_block_episodes else "Urban"
                state = env.reset(force_scenario=curr_scen)
                scen_hist.append(curr_scen)

                ep_r, ep_ho_fail, ep_pp, ep_ho, steps = 0.0, 0, 0, 0, 0
                done = False

                while not done:
                    st_time = time.perf_counter()
                    action = agent.select_action(state, is_training=True)
                    inference_latencies.append((time.perf_counter() - st_time) * 1000)

                    n_s, r, done, info = env.step(action)

                    # 训练逻辑保持不变：step reward 直接进 buffer
                    agent.store_transition(state, action, r, n_s, done)
                    agent.train_step()

                    state = n_s
                    ep_r += float(r)

                    # KPI 统计（与你原 02 一致）
                    ep_ho_fail += int(info.get('is_rlf', 0))
                    ep_pp += int(info.get('is_pingpong', 0))
                    ep_ho += int(info.get('ho_attempted', 0))
                    steps += 1

                # 计算 KPI（与你原 02 一致）
                hfr = ep_ho_fail / max(ep_ho, 1)
                ppr = ep_pp / (ep_ho + 1e-6)
                ehr = 1.0 - hfr - ppr

                r_hist.append(ep_r)
                hfr_hist.append(hfr)
                ppr_hist.append(ppr)
                ehr_hist.append(ehr)
                rlf_steps_hist.append(ep_ho_fail)
                steps_hist.append(steps)
                pingpong_hist.append(ep_pp)
                handover_hist.append(ep_ho)
                if curr_scen == "Highway":
                    h_r_hist.append(ep_r)
                    h_hfr_hist.append(hfr)
                    h_ppr_hist.append(ppr)
                    h_ehr_hist.append(ehr)
                else:
                    u_r_hist.append(ep_r)
                    u_hfr_hist.append(hfr)
                    u_ppr_hist.append(ppr)
                    u_ehr_hist.append(ehr)

                if ep % 10 == 0:
                    agent.update_target_network()

            # 存储该 seed 的完整序列（长度都严格等于 MAX_EPISODES）
            all_r.append(r_hist)
            all_hfr.append(hfr_hist)
            all_ppr.append(ppr_hist)
            all_ehr.append(ehr_hist)
            all_scen.append(scen_hist)
            all_rlf_steps.append(rlf_steps_hist)
            all_steps.append(steps_hist)
            all_pingpong.append(pingpong_hist)
            all_handover.append(handover_hist)
            all_h_r.append(h_r_hist)
            all_h_hfr.append(h_hfr_hist)
            all_h_ppr.append(h_ppr_hist)
            all_h_ehr.append(h_ehr_hist)
            all_u_r.append(u_r_hist)
            all_u_hfr.append(u_hfr_hist)
            all_u_ppr.append(u_ppr_hist)
            all_u_ehr.append(u_ehr_hist)

            # 断点保护：每个 seed 一个 npz
            np.savez(
                f"{save_dir}/raw_{name.replace(' ', '_')}_seed{seed}.npz",
                r=np.array(r_hist, dtype=float),
                hfr=np.array(hfr_hist, dtype=float),
                ppr=np.array(ppr_hist, dtype=float),
                ehr=np.array(ehr_hist, dtype=float),
                rlf_steps=np.array(rlf_steps_hist, dtype=float),
                steps=np.array(steps_hist, dtype=float),
                pingpong=np.array(pingpong_hist, dtype=float),
                handover=np.array(handover_hist, dtype=float),
                scen=np.array(scen_hist),
                h_r=np.array(h_r_hist, dtype=float),
                h_hfr=np.array(h_hfr_hist, dtype=float),
                h_ppr=np.array(h_ppr_hist, dtype=float),
                h_ehr=np.array(h_ehr_hist, dtype=float),
                u_r=np.array(u_r_hist, dtype=float),
                u_hfr=np.array(u_hfr_hist, dtype=float),
                u_ppr=np.array(u_ppr_hist, dtype=float),
                u_ehr=np.array(u_ehr_hist, dtype=float)
            )

        # 写入 raw_evidence（与新 03 完全匹配的 key 命名）
        raw_evidence[f"{name}_r"] = np.array(all_r, dtype=float)       # (S, E)
        raw_evidence[f"{name}_hfr"] = np.array(all_hfr, dtype=float)   # (S, E)
        raw_evidence[f"{name}_ppr"] = np.array(all_ppr, dtype=float)   # (S, E)
        raw_evidence[f"{name}_ehr"] = np.array(all_ehr, dtype=float)   # (S, E)
        raw_evidence[f"{name}_scen"] = np.array(all_scen)              # (S, E) strings
        raw_evidence[f"{name}_rlf_steps"] = np.array(all_rlf_steps, dtype=float)  # (S, E)
        raw_evidence[f"{name}_steps"] = np.array(all_steps, dtype=float)           # (S, E)
        raw_evidence[f"{name}_pingpong"] = np.array(all_pingpong, dtype=float)     # (S, E)
        raw_evidence[f"{name}_handover"] = np.array(all_handover, dtype=float)     # (S, E)
        raw_evidence[f"{name}_h_r"] = np.array(all_h_r, dtype=object)      # ragged by seed
        raw_evidence[f"{name}_h_hfr"] = np.array(all_h_hfr, dtype=object)  # ragged by seed
        raw_evidence[f"{name}_h_ppr"] = np.array(all_h_ppr, dtype=object)  # ragged by seed
        raw_evidence[f"{name}_h_ehr"] = np.array(all_h_ehr, dtype=object)  # ragged by seed
        raw_evidence[f"{name}_u_r"] = np.array(all_u_r, dtype=object)      # ragged by seed
        raw_evidence[f"{name}_u_hfr"] = np.array(all_u_hfr, dtype=object)  # ragged by seed
        raw_evidence[f"{name}_u_ppr"] = np.array(all_u_ppr, dtype=object)  # ragged by seed
        raw_evidence[f"{name}_u_ehr"] = np.array(all_u_ehr, dtype=object)  # ragged by seed
        model_data = {
            "h_ehr": np.array(all_h_ehr, dtype=object),
            "u_ehr": np.array(all_u_ehr, dtype=object),
        }
        raw_evidence[f"{name}_h_ehr"] = model_data["h_ehr"]
        raw_evidence[f"{name}_u_ehr"] = model_data["u_ehr"]

        # Leaderboard??????????????????
        def tail_mean(mat_2d, tail=100):
            mat_2d = np.asarray(mat_2d, dtype=float)
            return float(np.mean(mat_2d[:, -tail:]))

        # Highway / Urban Score?? scen mask ?????????
        def scen_tail_mean(values_2d, scen_2d, target_scene, tail=50):
            values_2d = np.asarray(values_2d, dtype=float)
            scen_2d = np.asarray(scen_2d)
            S, E = values_2d.shape
            start = max(0, E - tail)
            vals = []
            for s in range(S):
                mask = (scen_2d[s, start:E] == target_scene)
                if np.any(mask):
                    vals.append(np.mean(values_2d[s, start:E][mask]))
                else:
                    # fallback: if the tail window has no such scenario, use full-episode mean for that scenario
                    full_mask = (scen_2d[s, :] == target_scene)
                    if np.any(full_mask):
                        vals.append(np.mean(values_2d[s, :][full_mask]))
            return float(np.mean(vals)) if len(vals) > 0 else np.nan

        def ragged_tail_mean(obj_arr, tail=50):
            arr = np.asarray(obj_arr, dtype=object)
            vals = []
            for seq in arr:
                seq = np.asarray(seq, dtype=float)
                if seq.size > 0:
                    vals.append(float(np.mean(seq[-tail:])))
            return float(np.mean(vals)) if vals else np.nan

        # For Fig05, align AI score definition with Stage1 baseline envelope:
        # use scene-specific reward series (h_r/u_r) when available,
        # fallback to old scenario-mask method for compatibility.
        if f"{name}_h_r" in raw_evidence and f"{name}_u_r" in raw_evidence:
            highway_score = ragged_tail_mean(raw_evidence[f"{name}_h_r"], tail=50)
            urban_score = ragged_tail_mean(raw_evidence[f"{name}_u_r"], tail=50)
        else:
            highway_score = scen_tail_mean(raw_evidence[f"{name}_r"], raw_evidence[f"{name}_scen"], "Highway", tail=50)
            urban_score = scen_tail_mean(raw_evidence[f"{name}_r"], raw_evidence[f"{name}_scen"], "Urban", tail=50)

        final_report.append({
            "Model": name,
            "Mean Score": tail_mean(raw_evidence[f"{name}_r"], scenario_cycle_episodes),
            "EHR": tail_mean(raw_evidence[f"{name}_ehr"], scenario_cycle_episodes),
            "HFR": tail_mean(raw_evidence[f"{name}_hfr"], scenario_cycle_episodes),
            "PPR": tail_mean(raw_evidence[f"{name}_ppr"], scenario_cycle_episodes),
            "Latency": float(np.mean(inference_latencies)),
            "Highway Score": highway_score,
            "Urban Score": urban_score,
        })

        pd.DataFrame(final_report).to_csv(f"{save_dir}/final_leaderboard.csv", index=False)

    np.savez(f"{save_dir}/raw_evidence.npz", **raw_evidence)
    print(f"🏁 Training Complete. Data saved in {save_dir}. Now run 03 and 04.")

if __name__ == "__main__":
    train()
