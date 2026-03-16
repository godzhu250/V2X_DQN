import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch

import config
from dqn_agent import DQNAgent
from lstm_dqn_agent import LSTMDQNAgent
from vehicle_env import VehicleEnv


SCENARIO = "Urban"
MODEL_SPECS = [
    ("DQN (MLP)", "DQN_MLP", DQNAgent),
    ("DRQN (LSTM)", "DRQN_LSTM", LSTMDQNAgent),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=config.URBAN_EPISODES)
    parser.add_argument("--seeds", type=str, default=",".join(map(str, config.URBAN_SEEDS)))
    return parser.parse_args()


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_seeds(seed_str):
    return [int(x.strip()) for x in seed_str.split(",") if x.strip()]


def run_train_episode(env, agent):
    if hasattr(agent, "state_sequence"):
        agent.state_sequence.clear()

    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.select_action(state, is_training=True)
        next_state, reward, done, _ = env.step(action)

        agent.store_transition(state, action, reward, next_state, done)
        agent.train_step()

        state = next_state
        total_reward += float(reward)

    stats = env.get_episode_stats()
    return {
        "reward": float(total_reward),
        "ho_attempted": int(stats["ho_attempted"]),
        "ho_success": int(stats["ho_success"]),
        "ho_failed": int(stats["ho_failed"]),
        "pingpong": int(stats["pingpong"]),
        "weak_signal_event": int(stats["weak_signal_event"]),
        "hfr": float(stats["hfr"]),
        "ppr": float(stats["ppr"]),
        "ehr": float(stats["ehr"]),
    }


def summarize_seed(df, scenario, model_name, model_tag, seed):
    total_attempted = int(df["ho_attempted"].sum())
    total_success = int(df["ho_success"].sum())
    total_failed = int(df["ho_failed"].sum())
    total_pingpong = int(df["pingpong"].sum())
    total_weak_signal = int(df["weak_signal_event"].sum())

    agg_hfr = total_failed / max(total_attempted, 1)
    agg_ppr = total_pingpong / max(total_attempted, 1)
    agg_ehr = 1.0 - agg_hfr - agg_ppr

    tail_n = min(100, len(df))
    tail_df = df.tail(tail_n)

    return {
        "scenario": scenario,
        "model": model_name,
        "model_tag": model_tag,
        "seed": int(seed),
        "episodes": int(len(df)),
        "mean_reward": float(df["reward"].mean()),
        "mean_hfr": float(df["hfr"].mean()),
        "mean_ppr": float(df["ppr"].mean()),
        "mean_ehr": float(df["ehr"].mean()),
        "tail100_reward": float(tail_df["reward"].mean()),
        "tail100_hfr": float(tail_df["hfr"].mean()),
        "tail100_ppr": float(tail_df["ppr"].mean()),
        "tail100_ehr": float(tail_df["ehr"].mean()),
        "aggregate_hfr": float(agg_hfr),
        "aggregate_ppr": float(agg_ppr),
        "aggregate_ehr": float(agg_ehr),
        "total_ho_attempted": total_attempted,
        "total_ho_success": total_success,
        "total_ho_failed": total_failed,
        "total_pingpong": total_pingpong,
        "total_weak_signal_event": total_weak_signal,
    }


def summarize_model(seed_summary_df, scenario, model_name, model_tag):
    total_attempted = int(seed_summary_df["total_ho_attempted"].sum())
    total_failed = int(seed_summary_df["total_ho_failed"].sum())
    total_pingpong = int(seed_summary_df["total_pingpong"].sum())

    agg_hfr = total_failed / max(total_attempted, 1)
    agg_ppr = total_pingpong / max(total_attempted, 1)
    agg_ehr = 1.0 - agg_hfr - agg_ppr

    return {
        "scenario": scenario,
        "model": model_name,
        "model_tag": model_tag,
        "n_seeds": int(len(seed_summary_df)),
        "episodes_per_seed": int(seed_summary_df["episodes"].iloc[0]),
        "mean_reward": float(seed_summary_df["mean_reward"].mean()),
        "mean_hfr": float(seed_summary_df["mean_hfr"].mean()),
        "mean_ppr": float(seed_summary_df["mean_ppr"].mean()),
        "mean_ehr": float(seed_summary_df["mean_ehr"].mean()),
        "tail100_reward": float(seed_summary_df["tail100_reward"].mean()),
        "tail100_hfr": float(seed_summary_df["tail100_hfr"].mean()),
        "tail100_ppr": float(seed_summary_df["tail100_ppr"].mean()),
        "tail100_ehr": float(seed_summary_df["tail100_ehr"].mean()),
        "aggregate_hfr": float(agg_hfr),
        "aggregate_ppr": float(agg_ppr),
        "aggregate_ehr": float(agg_ehr),
        "total_ho_attempted": total_attempted,
        "total_ho_failed": total_failed,
        "total_pingpong": total_pingpong,
    }


def main():
    args = parse_args()
    seeds = parse_seeds(args.seeds)

    out_dir = os.path.join("Result", "urban", "train")
    model_dir = os.path.join(out_dir, "models")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    run_meta = {
        "scenario": SCENARIO,
        "episodes": int(args.episodes),
        "seeds": seeds,
        "episode_steps": int(config.EPISODE_STEPS),
        "sim_step_seconds": float(config.SIM_STEP_SECONDS),
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    all_seed_rows = []
    all_model_rows = []

    for model_name, model_tag, agent_cls in MODEL_SPECS:
        print(f"[Train][{SCENARIO}] model={model_name}")
        model_seed_rows = []

        for seed in seeds:
            set_global_seed(seed)

            env = VehicleEnv(scenario=SCENARIO)
            agent = agent_cls()

            episode_rows = []
            for ep in range(1, args.episodes + 1):
                row = run_train_episode(env, agent)
                row["episode"] = ep
                episode_rows.append(row)

                if ep % config.TARGET_UPDATE_INTERVAL == 0:
                    agent.update_target_network()

                if ep % 200 == 0 or ep == 1 or ep == args.episodes:
                    print(
                        f"  seed={seed} ep={ep}/{args.episodes} "
                        f"reward={row['reward']:.3f} hfr={row['hfr']:.4f} "
                        f"ppr={row['ppr']:.4f} ehr={row['ehr']:.4f}"
                    )

            ep_df = pd.DataFrame(episode_rows).sort_values("episode").reset_index(drop=True)
            ep_path = os.path.join(out_dir, f"{model_tag}_seed{seed}.csv")
            ep_df.to_csv(ep_path, index=False)

            model_path = os.path.join(model_dir, f"{model_tag}_seed{seed}.pt")
            agent.save_model(model_path)

            seed_row = summarize_seed(ep_df, SCENARIO, model_name, model_tag, seed)
            model_seed_rows.append(seed_row)
            all_seed_rows.append(seed_row)

            print(
                f"  saved seed={seed} -> {ep_path}, "
                f"aggregate_hfr={seed_row['aggregate_hfr']:.4f}, "
                f"aggregate_ppr={seed_row['aggregate_ppr']:.4f}, "
                f"aggregate_ehr={seed_row['aggregate_ehr']:.4f}"
            )

        model_seed_df = pd.DataFrame(model_seed_rows)
        model_summary_row = summarize_model(model_seed_df, SCENARIO, model_name, model_tag)
        all_model_rows.append(model_summary_row)

    seed_summary_df = pd.DataFrame(all_seed_rows)
    seed_summary_df.to_csv(os.path.join(out_dir, "seed_summary.csv"), index=False)

    model_summary_df = pd.DataFrame(all_model_rows)
    model_summary_df.to_csv(os.path.join(out_dir, "model_summary.csv"), index=False)

    print(f"Saved: {os.path.join(out_dir, 'seed_summary.csv')}")
    print(f"Saved: {os.path.join(out_dir, 'model_summary.csv')}")


if __name__ == "__main__":
    main()
