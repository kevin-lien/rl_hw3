"""Grade a submitted 2048 model. DON'T MODIFY: the TAs grade with the original version of this file.

    python eval_2048.py --model path/to/model_2048.zip

Plays 100 games (deterministic actions) and records the highest tile of each.
A baseline is passed if at least one of the 100 games reaches its tile, and each
baseline is worth 5 points on this seed set, so this script reports up to 15:
    simple: 128 -> 5,   medium: 256 -> 5,   strong: 512 -> 5
Grading runs it twice: on the public seeds (the default, 0-99) and on 100
TA-chosen private seeds (--seed-base), 15 + 15 = 30 points in total. Each run is
limited to 10 minutes; a run that takes longer scores 0.
"""
import argparse
import time

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import DQN

import numpy as np
from collections import Counter

register(
    id='2048-eval',
    entry_point='envs:Eval2048Env'
)

BASELINES = [("simple", 128), ("medium", 256), ("strong", 512)]
POINTS_PER_BASELINE = 5
TIME_LIMIT_S = 10 * 60


def evaluation(env, model, render_last, eval_num=100, seed_base=0, time_limit=None):
    score = []
    highest = []
    start = time.time()

    ### Run eval_num times rollouts
    for seed in range(seed_base, seed_base + eval_num):
        if time_limit is not None and time.time() - start > time_limit:
            return score, highest, True
        done = False
        # Set seed and reset env using Gymnasium API
        obs, info = env.reset(seed=seed)

        while not done:
            # Interact with env using Gymnasium API
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)

        # Render the last board state of each episode
        # print("Last board state:")
        # env.render()

        score.append(info['score'])
        highest.append(info['highest'])

    ### Render last rollout
    if render_last:
        print("Rendering last rollout")
        done = False
        obs, info = env.reset(seed=seed_base + eval_num - 1)
        env.render()

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            env.render()


    return score, highest, False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/sample_model", help="path to your SB3 .zip")
    parser.add_argument("--seed-base", type=int, default=0, help="seeds are seed_base .. seed_base+99")
    args = parser.parse_args()
    model_path = args.model
    env = gym.make('2048-eval')

    ### Load model with SB3
    # 2048 submissions must be DQN models (DoubleDQN from train_2048.py saves as one too).
    try:
        model = DQN.load(model_path)
    except Exception as e:
        raise ValueError(f"Could not load {model_path} as a DQN model: {e}") from e
    print("Successfully loaded model using DQN")

    eval_num = 100
    score, highest, timed_out = evaluation(env, model, False, eval_num, args.seed_base, TIME_LIMIT_S)
    if timed_out:
        print(f"Time limit exceeded ({TIME_LIMIT_S // 60} min) after {len(highest)} rollouts")
        print(f"Score: 0/{POINTS_PER_BASELINE * len(BASELINES)}")
        raise SystemExit(1)

    print("Avg_score:  ", np.sum(score)/eval_num)
    print("Avg_highest:", np.sum(highest)/eval_num)


    print(f"Counts: (Total of {eval_num} rollouts)")
    c = Counter(highest)
    for item in (sorted(c.items(),key = lambda i: i[0])):
        print(f"{item[0]}: {item[1]}")

    best = max(highest)
    total = 0
    print(f"\nBaselines (best tile over {eval_num} rollouts = {best}, seeds {args.seed_base}-{args.seed_base + eval_num - 1}):")
    for name, tile in BASELINES:
        points = POINTS_PER_BASELINE if best >= tile else 0
        total += points
        print(f"  {name:6s} (reach {tile:3d} in at least one rollout): {points}/{POINTS_PER_BASELINE}")
    print(f"Score: {total}/{POINTS_PER_BASELINE * len(BASELINES)}")
