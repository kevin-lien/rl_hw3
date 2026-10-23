"""Grade a submitted Astro Party model against the two TA bots.
DON'T MODIFY: the TAs grade with the original version of this file.

    python eval_astro_party.py --model path/to/model_astro_party.zip

Your model plays 100 rollouts against each TA bot on AstroPartyEvalEnv, with
deterministic actions:
  - medium baseline: rusher_bot (models/rusher_bot.py), rule-based
  - strong baseline: sniper_bot (models/sniper_bot.py), rule-based
A rollout counts as won only if your ship is the one left alive; a draw at the
1500-step time limit is not a win. Reaching the baseline's win rate earns its
points on this seed set: 90% against rusher_bot (medium) and 85% against
sniper_bot (strong), 5 points each, so up to 10 here.
Grading runs it twice: on the public seeds (the default, 0-99) and on 100
TA-chosen private seeds (--seed-base), 10 + 10 = 20 points in total.
"""
import argparse

import numpy as np
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from envs.eval_astro_party_env import AstroPartyEvalEnv
from models.rusher_bot import rusher_bot_action
from models.sniper_bot import sniper_bot_action

NUM_ROLLOUTS = 100
WIN_RATE_TO_PASS = {"medium": 0.90, "strong": 0.85}
POINTS = {"medium": 5, "strong": 5}


def decode_action(flat_action) -> dict:
    """Box(3,) [turn_rate, thrust, shoot_signal] -> AstroPartyEvalEnv's native
    per-ship action. Both the submitted model's prediction and the baseline's
    output go through this -- same uniform action representation."""
    move = np.clip(np.asarray(flat_action[:2], dtype=np.float32), -1.0, 1.0)
    shoot = int(flat_action[2] > 0.0)
    return {"move": move, "shoot": shoot}


def load_model(model_path):
    # Note: Model can be loaded with arbitrary algorithm class for evaluation.
    # The action space is a continuous Box(3,), which rules out DQN
    # (discrete-only) -- unlike 2048's eval_2048.py, it isn't in this try-list.
    for model_class in [PPO, A2C, SAC, TD3, DDPG]:
        try:
            model = model_class.load(model_path, device="cpu")
            print(f"Loaded {model_path} with {model_class.__name__}")
            return model
        except Exception:
            continue
    raise ValueError(f"Could not load model from {model_path} with any of the available algorithms")


def play(model, opponent_fn, seed, render=False):
    """One duel of model against opponent_fn. Returns (won, steps)."""
    env = AstroPartyEvalEnv(render_mode="human" if render else None)
    obs, info = env.reset(seed=seed)
    done, steps = False, 0
    while not done:
        action, _state = model.predict(obs["ship1"], deterministic=True)
        opponent_action = opponent_fn(obs["ship2"])
        obs, rewards, terms, truncs, info = env.step({
            "ship1": decode_action(action),
            "ship2": decode_action(opponent_action),
        })
        steps += 1
        done = terms["ship1"] or truncs["ship1"]
    won = env.ships["ship1"].alive and not env.ships["ship2"].alive
    env.close()
    return won, steps


def evaluation(model, opponent_fn, seeds, render_last=False):
    results = []
    for i, seed in enumerate(seeds):
        results.append(play(model, opponent_fn, seed, render=render_last and i == len(seeds) - 1))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/sample_model", help="path to your SB3 .zip")
    parser.add_argument("--seed-base", type=int, default=0, help="seeds are seed_base .. seed_base+99")
    parser.add_argument("--render", action="store_true", help="watch the last duel against each baseline")
    args = parser.parse_args()

    model = load_model(args.model)
    baselines = {
        "medium": ("rusher_bot", rusher_bot_action),
        "strong": ("sniper_bot", sniper_bot_action),
    }
    seeds = list(range(args.seed_base, args.seed_base + NUM_ROLLOUTS))

    total = 0
    for level, (name, opponent_fn) in baselines.items():
        results = evaluation(model, opponent_fn, seeds, args.render)
        wins = sum(won for won, _ in results)
        win_rate = wins / len(seeds)
        points = POINTS[level] if win_rate >= WIN_RATE_TO_PASS[level] else 0
        total += points
        print(f"\n{level} baseline ({name}), seeds {seeds[0]}-{seeds[-1]}:")
        print("  " + "".join("W" if won else "." for won, _ in results) + "   (W = win, . = loss or draw)")
        print(f"  win rate {win_rate:.0%} ({wins}/{len(seeds)}), need {WIN_RATE_TO_PASS[level]:.0%} -> {points}/{POINTS[level]}")

    print(f"\nScore: {total}/{sum(POINTS.values())}")
