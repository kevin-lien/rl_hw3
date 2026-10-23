"""Play 2048 alongside a saved DQN or DoubleDQN model.

From the 2048 folder: python try2.py --model models/model_2048.zip
DoubleDQN checkpoints load with DQN because their action selection is the same.
"""

import argparse

from stable_baselines3 import DQN
import numpy as np
import torch as th
from train_2048 import DoubleDQN

from envs.eval_2048_env import Eval2048Env


HUMAN_ACTIONS = {"w": 0, "d": 1, "s": 2, "a": 3}
ACTION_NAMES = ("up", "right", "down", "left")


def model_action(model, observation):
    action, _ = model.predict(observation, deterministic=True)
    return int(action)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model_2048.zip")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = Eval2048Env()
    model = DoubleDQN.load(args.model, device="cpu")
    model.double_q = True

    boards = [env.reset(seed=args.seed + i)[0] for i in range(3)]
    batch = th.as_tensor(np.stack(boards), device=model.device)

    with th.no_grad():
        print("Online values:", model.q_net(batch))
        print("Target values:", model.q_net_target(batch))
        print("Selected next values:", model.next_q_values(batch))

    env.close()
    


if __name__ == "__main__":
    main()
