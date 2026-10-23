"""Play the assignment's 2048 training environment in a terminal.

Run from the 2048 folder: python try.py
Use --seed N to replay the same starting board and tile sequence.
"""

import argparse

from envs.train_2048_env import My2048Env


ACTIONS = {"w": 0, "d": 1, "s": 2, "a": 3}


def main():
    parser = argparse.ArgumentParser(description="Play the 2048 training environment")
    parser.add_argument("--seed", type=int, help="repeatable game seed")
    args = parser.parse_args()

    env = My2048Env()
    try:
        while True:
            env.reset(seed=args.seed)
            print("\nNew game! Move with W/A/S/D; R restarts; Q quits.")
            print("An invalid move ends the game, just as it does in grading.")

            while True:
                print()
                env.render()
                try:
                    command = input("Move [W/A/S/D, R, Q]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    return

                if command == "q":
                    return
                if command == "r":
                    break
                if command not in ACTIONS:
                    print("Enter W, A, S, D, R, or Q.")
                    continue

                _, reward, terminated, truncated, info = env.step(ACTIONS[command])
                print(f"Move reward: {reward:g}")
                if info["illegal_move"]:
                    print("That move did not change the board, so the game ended.")
                if terminated or truncated:
                    print()
                    env.render()
                    print("Game over. Press R for another game or Q to quit.")
                    while True:
                        try:
                            choice = input("[R/Q]: ").strip().lower()
                        except (EOFError, KeyboardInterrupt):
                            print("\nGoodbye!")
                            return
                        if choice == "q":
                            return
                        if choice == "r":
                            break
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()



