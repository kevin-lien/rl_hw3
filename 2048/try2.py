"""Inspect batch sizes without training or changing a checkpoint.

From the 2048 folder: python try2.py
Or choose a checkpoint: python try2.py --model models/model_2048.zip
"""

import argparse
import inspect

from stable_baselines3 import DQN
from train_2048 import DoubleDQN, my_config


def main():
    parser = argparse.ArgumentParser(description="Check DQN batch sizes")
    parser.add_argument("--model", default=my_config["save_path"])
    args = parser.parse_args()

    default_batch = inspect.signature(DQN.__init__).parameters["batch_size"].default
    configured_batch = my_config["algorithm_kwargs"].get("batch_size", default_batch)
    method_default = inspect.signature(DoubleDQN.train).parameters["batch_size"].default

    print(f"SB3 DQN default batch size: {default_batch}")
    print(f"Batch size for a NEW run with your current config: {configured_batch}")
    print(f"DoubleDQN.train() fallback argument: {method_default}")
    print("SB3 passes model.batch_size to train(), overriding that fallback.")

    try:
        model = DQN.load(args.model, device="cpu")
    except FileNotFoundError:
        print(f"\nCheckpoint not found: {args.model}")
        print("The default and current config checks above still apply.")
    else:
        print(f"\nCheckpoint: {args.model}")
        print(f"Batch size saved in this checkpoint: {model.batch_size}")
        print("Editing the config does not change an existing checkpoint.")


if __name__ == "__main__":
    main()
