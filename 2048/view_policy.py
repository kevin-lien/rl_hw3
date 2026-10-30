"""Watch a saved policy in the original grading environment.

Run inside 2048:
    python view_policy.py --model models/model_2048_200k.zip --seed 0
    python view_policy.py --seed 0 --step

The policy's actions are never filtered or replaced with legal alternatives.
"""

import argparse
import time

import torch as th
from stable_baselines3 import DQN

from envs.eval_2048_env import Eval2048Env, IllegalMove


ACTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")


def print_board(env):
    border = "+" + "+".join(["-------"] * 4) + "+"
    print(border)
    for row in env.get_board():
        print("|" + "|".join(f"{int(tile) if tile else '.':^7}" for tile in row) + "|")
        print(border)


def legal_actions(env):
    """Trial moves inspect legality without changing the board or RNG."""
    actions = []
    for action in range(4):
        try:
            env.move(action, trial=True)
        except IllegalMove:
            continue
        actions.append(action)
    return actions


def watch_episode(env, model, seed, args):
    observation, _ = env.reset(seed=seed)
    print(f"\n{'=' * 60}\nSeed {seed} -- deterministic policy")
    print_board(env)

    for step in range(1, args.max_steps + 1):
        action_array, _ = model.predict(observation, deterministic=True)
        action = int(action_array.item())
        with th.no_grad():
            tensor, _ = model.policy.obs_to_tensor(observation)
            q_values = model.q_net(tensor)[0].cpu().tolist()
        legal = legal_actions(env)

        print(f"\nMove {step} | score {env.score:g} | highest {int(env.highest())}")
        print("Action       Q-value       Status")
        for index, value in enumerate(q_values):
            marker = " <-- POLICY CHOICE" if index == action else ""
            status = "legal" if index in legal else "ILLEGAL"
            print(f"{ACTION_NAMES[index]:<9} {value:>10.3f}       {status}{marker}")

        if args.step:
            command = input("Enter: execute this move | q: quit > ").strip().lower()
            if command == "q":
                return False
        elif args.delay:
            time.sleep(args.delay)

        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Executed {ACTION_NAMES[action]} | move reward {reward:g}")
        print_board(env)

        if terminated or truncated:
            print(f"\nGAME OVER -- seed {seed}, move {step}")
            print(f"Final score: {info['score']:g} | highest tile: {int(info['highest'])}")
            if info["illegal_move"]:
                print(f"Cause: policy chose {ACTION_NAMES[action]}, which did not change the board.")
                if legal:
                    alternatives = ", ".join(ACTION_NAMES[index] for index in legal)
                    print(f"Legal alternatives on that board: {alternatives}")
                    print("The game could have continued with a different action.")
                else:
                    print("No legal moves were available.")
            elif truncated:
                print("Cause: environment truncated the episode.")
            elif not legal_actions(env):
                print("Cause: no legal moves remain after the last move and new tile.")
            else:
                print("Cause: the environment's terminal condition was reached.")
            print("Replay this seed with --step to inspect the decisions.")
            return True

    print(f"Viewer stopped after {args.max_steps} moves; this is a viewer limit, not a game death.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Watch a 2048 model and diagnose game endings")
    parser.add_argument("--model", default="models/model_2048_200k.zip")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1, help="play consecutive seeds")
    parser.add_argument("--delay", type=float, default=0.35, help="seconds before each move")
    parser.add_argument("--step", action="store_true", help="press Enter before every move")
    parser.add_argument("--max-steps", type=int, default=10000)
    args = parser.parse_args()
    if args.delay < 0 or args.episodes < 1 or args.max_steps < 1:
        parser.error("delay must be nonnegative; episodes and max-steps must be positive")

    model = DQN.load(args.model, device="cpu")
    print(f"Loaded {args.model}")
    print("Using grading rules: an unchanged-board move ends the game.")
    print("Q-values estimate future reward; they are not action probabilities.")
    print("Ctrl+C quits. Terminal history preserves previous boards.")
    env = Eval2048Env()
    try:
        for seed in range(args.seed, args.seed + args.episodes):
            if not watch_episode(env, model, seed, args):
                break
    except (KeyboardInterrupt, EOFError):
        print("\nViewer closed.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
