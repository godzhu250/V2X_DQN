import argparse
import importlib.util
import os

import config


def load_train_core():
    path = os.path.join(os.path.dirname(__file__), "02_main.py")
    spec = importlib.util.spec_from_file_location("train_core_02_main", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Failed to load 02_main.py")
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=config.HIGHWAY_EPISODES)
    parser.add_argument("--seeds", type=str, default=",".join(map(str, config.HIGHWAY_SEEDS)))
    return parser.parse_args()


def parse_seeds(seed_str):
    return [int(x.strip()) for x in seed_str.split(",") if x.strip()]


def main():
    args = parse_args()
    seeds = parse_seeds(args.seeds)

    train_core = load_train_core()
    train_core.run_training(
        scenario="Highway",
        seeds=seeds,
        episodes=int(args.episodes),
        output_dir=os.path.join("Result", "highway", "train"),
    )


if __name__ == "__main__":
    main()
