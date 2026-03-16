import argparse
import gc
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

import config
from dqn_agent import DQNAgent
from lstm_dqn_agent import LSTMDQNAgent
from vehicle_env import VehicleEnv


MODEL_SPECS = [
    ("DQN (MLP)", "DQN_MLP", DQNAgent),
    ("LSTM-DQN", "LSTM_DQN", LSTMDQNAgent),
]


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_train_episode(env, agent, ep=None):
    if hasattr(agent, "state_sequence"):
        agent.state_sequence.clear()

    state = env.reset()
    done = False
    total_reward = 0.0

    step = 0
    while not done:
        if step == 0 and ep is not None and ep % 200 == 0:
            print(f"[Debug] Training frequency = every {config.TRAIN_FREQ} steps")

        action = agent.select_action(state, is_training=True)
        next_state, reward, done, _ = env.step(action)

        agent.store_transition(state, action, reward, next_state, done)
        if step % config.TRAIN_FREQ == 0:
            agent.train_step()

        state = next_state
        total_reward += float(reward)
        step += 1

    stats = env.get_episode_stats()
    return {
        "reward": float(total_reward),
        "ho_attempted": int(stats["ho_attempted"]),
        "ho_success": int(stats["ho_success"]),
        "ho_failed": int(stats["ho_failed"]),
        "pingpong": int(stats["pingpong"]),
        "weak_signal_event": int(stats["weak_signal_event"]),
        "no_attempt_episode": int(stats["no_attempt_episode"]),
        "pending_validation_started": int(stats["pending_validation_started"]),
        "validation_success": int(stats["validation_success"]),
        "validation_failure": int(stats["validation_failure"]),
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
    no_attempt_episode_count = int(df["no_attempt_episode"].sum())
    total_pending_validation_started = int(df["pending_validation_started"].sum())
    total_validation_success = int(df["validation_success"].sum())
    total_validation_failure = int(df["validation_failure"].sum())

    aggregate_hfr = total_failed / max(total_attempted, 1)
    aggregate_ppr = total_pingpong / max(total_attempted, 1)
    aggregate_ehr = 1.0 - aggregate_hfr - aggregate_ppr

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
        "aggregate_hfr": float(aggregate_hfr),
        "aggregate_ppr": float(aggregate_ppr),
        "aggregate_ehr": float(aggregate_ehr),
        "total_ho_attempted": total_attempted,
        "total_ho_success": total_success,
        "total_ho_failed": total_failed,
        "total_pingpong": total_pingpong,
        "total_weak_signal_event": total_weak_signal,
        "no_attempt_episode_count": no_attempt_episode_count,
        "no_attempt_episode_ratio": float(no_attempt_episode_count / max(len(df), 1)),
        "total_pending_validation_started": total_pending_validation_started,
        "total_validation_success": total_validation_success,
        "total_validation_failure": total_validation_failure,
    }


def summarize_model(seed_summary_df, scenario, model_name, model_tag):
    total_attempted = int(seed_summary_df["total_ho_attempted"].sum())
    total_success = int(seed_summary_df["total_ho_success"].sum())
    total_failed = int(seed_summary_df["total_ho_failed"].sum())
    total_pingpong = int(seed_summary_df["total_pingpong"].sum())
    total_weak_signal = int(seed_summary_df["total_weak_signal_event"].sum())
    no_attempt_episode_count = int(seed_summary_df["no_attempt_episode_count"].sum())
    total_episodes = int(seed_summary_df["episodes"].sum())
    total_pending_validation_started = int(seed_summary_df["total_pending_validation_started"].sum())
    total_validation_success = int(seed_summary_df["total_validation_success"].sum())
    total_validation_failure = int(seed_summary_df["total_validation_failure"].sum())

    aggregate_hfr = total_failed / max(total_attempted, 1)
    aggregate_ppr = total_pingpong / max(total_attempted, 1)
    aggregate_ehr = 1.0 - aggregate_hfr - aggregate_ppr

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
        "aggregate_hfr": float(aggregate_hfr),
        "aggregate_ppr": float(aggregate_ppr),
        "aggregate_ehr": float(aggregate_ehr),
        "total_ho_attempted": total_attempted,
        "total_ho_success": total_success,
        "total_ho_failed": total_failed,
        "total_pingpong": total_pingpong,
        "total_weak_signal_event": total_weak_signal,
        "no_attempt_episode_count": no_attempt_episode_count,
        "no_attempt_episode_ratio": float(no_attempt_episode_count / max(total_episodes, 1)),
        "total_pending_validation_started": total_pending_validation_started,
        "total_validation_success": total_validation_success,
        "total_validation_failure": total_validation_failure,
    }


def run_training(scenario, seeds, episodes, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    torch.set_num_threads(4)

    run_meta = {
        "scenario": scenario,
        "episodes": int(episodes),
        "seeds": [int(s) for s in seeds],
        "episode_steps": int(config.EPISODE_STEPS),
        "sim_step_seconds": float(config.SIM_STEP_SECONDS),
    }
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    all_seed_rows = []
    all_model_rows = []

    for model_name, model_tag, agent_cls in MODEL_SPECS:
        print(f"[Train][{scenario}] model={model_name}")
        model_seed_rows = []

        for seed in seeds:
            set_global_seed(int(seed))

            env = VehicleEnv(scenario=scenario)
            agent = agent_cls()

            episode_rows = []
            for ep in range(1, int(episodes) + 1):
                row = run_train_episode(env, agent, ep=ep)
                row["episode"] = ep
                episode_rows.append(row)

                if ep % config.TARGET_UPDATE_INTERVAL == 0:
                    agent.update_target_network()

                if ep % 200 == 0 or ep == 1 or ep == int(episodes):
                    print(
                        f"  seed={seed} ep={ep}/{episodes} "
                        f"reward={row['reward']:.3f} hfr={row['hfr']:.4f} "
                        f"ppr={row['ppr']:.4f} ehr={row['ehr']:.4f}"
                    )

            ep_df = pd.DataFrame(episode_rows).sort_values("episode").reset_index(drop=True)
            ep_path = os.path.join(output_dir, f"{model_tag}_seed{seed}.csv")
            ep_df.to_csv(ep_path, index=False)

            # Seed-level robust checkpoint save.
            seed_file = os.path.join(output_dir, f"seed_{seed}_episodes.csv")
            df_seed = ep_df.copy()
            df_seed.insert(0, "model", model_name)
            file_exists = os.path.exists(seed_file)
            df_seed.to_csv(
                seed_file,
                mode="a" if file_exists else "w",
                header=not file_exists,
                index=False,
            )
            print(f"[Train] Seed {seed} results saved to {seed_file}")

            sys.stdout.flush()
            if hasattr(os, "sync"):
                try:
                    os.sync()
                except OSError:
                    pass

            model_path = os.path.join(model_dir, f"{model_tag}_seed{seed}.pt")
            agent.save_model(model_path)

            seed_row = summarize_seed(ep_df, scenario, model_name, model_tag, seed)
            model_seed_rows.append(seed_row)
            all_seed_rows.append(seed_row)

            print(
                f"  saved seed={seed} -> {ep_path}, "
                f"aggregate_hfr={seed_row['aggregate_hfr']:.4f}, "
                f"aggregate_ppr={seed_row['aggregate_ppr']:.4f}, "
                f"aggregate_ehr={seed_row['aggregate_ehr']:.4f}"
            )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            episode_rows = None
            print(f"[Train] Seed {seed} cleanup complete. Starting next seed...")

        model_seed_df = pd.DataFrame(model_seed_rows)
        model_summary_row = summarize_model(model_seed_df, scenario, model_name, model_tag)
        all_model_rows.append(model_summary_row)

    seed_summary_df = pd.DataFrame(all_seed_rows)
    seed_summary_df.to_csv(os.path.join(output_dir, "seed_summary.csv"), index=False)

    model_summary_df = pd.DataFrame(all_model_rows)
    model_summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)

    print(f"Saved: {os.path.join(output_dir, 'seed_summary.csv')}")
    print(f"Saved: {os.path.join(output_dir, 'model_summary.csv')}")


def _parse_seeds(seed_str):
    return [int(x.strip()) for x in seed_str.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, required=True, choices=["Highway", "Urban"])
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seeds", type=str, required=True, help="Comma-separated, e.g., 71,123,456")
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    run_training(
        scenario=args.scenario,
        seeds=seeds,
        episodes=args.episodes,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
