import time
import warnings

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from envs.train_astro_party_env import AstroPartyTrainEnv, decode_action
from envs.eval_astro_party_env import AstroPartyEvalEnv
from models.rusher_bot import rusher_bot_action

warnings.filterwarnings("ignore")

# Set hyper params (configurations) for training
my_config = {
    "run_id": "example",
    "algorithm": PPO,
    "policy_network": "MlpPolicy",
    "save_path": "models/model_astro_party",  # saved as models/model_astro_party.zip, the file you submit
    "num_train_envs": 4,
    "epoch_num": 5,
    "timesteps_per_epoch": 20_000,
    "eval_episode_num": 20,
}

# Who your opponent is during training is entirely up to you -- the rusher bot
# below (the medium baseline) is just the default. eval_astro_party.py grades you
# against models/rusher_bot.py (medium) and models/sniper_bot.py (strong).
# Point `opponent_fn` at anything with
# the same obs(84,) -> action(3,) signature: a different heuristic, a frozen
# snapshot of your own model that you reload yourself partway through
# training (self-play), a pool of past snapshots, etc. You can also ignore
# this file's structure entirely and write your own training loop -- the
# only thing that's graded is the model you end up saving to
# my_config["save_path"].
def make_env():
    return AstroPartyTrainEnv(opponent_fn=rusher_bot_action)


def evaluate(model, eval_episode_num):
    """Roll out eval_episode_num duels against the rusher bot using
    AstroPartyEvalEnv's sparse "dead astronaut" reward -- i.e. the same
    signal eval_astro_party.py actually grades with -- so this number is meaningful
    progress tracking even though training itself uses train_astro_party_env.py's dense
    shaped reward."""
    env = AstroPartyEvalEnv()
    wins, returns = 0, []
    for seed in range(eval_episode_num):
        obs, _ = env.reset(seed=seed)
        done = False
        ep_return = 0.0
        while not done:
            action, _ = model.predict(obs["ship1"], deterministic=True)
            opponent_action = rusher_bot_action(obs["ship2"])
            obs, rewards, terms, truncs, _ = env.step({
                "ship1": decode_action(action),
                "ship2": decode_action(opponent_action),
            })
            ep_return += rewards["ship1"]
            done = terms["ship1"] or truncs["ship1"]
        returns.append(ep_return)
        if env.ships["ship1"].alive and not env.ships["ship2"].alive:
            wins += 1
    env.close()
    return wins / eval_episode_num, float(np.mean(returns))


def train(model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best_win_rate = -1.0
    start_time = time.time()

    for epoch in range(config["epoch_num"]):
        epoch_start_time = time.time()

        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
        )

        epoch_duration = time.time() - epoch_start_time
        total_duration = time.time() - start_time

        ### Evaluation
        eval_start = time.time()
        win_rate, avg_return = evaluate(model, config["eval_episode_num"])
        eval_duration = time.time() - eval_start

        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['epoch_num']} completed")
        print(f"{'='*60}")
        print(f"Training Speed:")
        print(f"   - Epoch time: {epoch_duration:.1f}s")
        print(f"   - Eval time: {eval_duration:.1f}s")
        print(f"   - Total time: {total_duration/60:.1f} min")
        print(f"Performance (vs rusher bot, sparse eval reward):")
        print(f"   - Win rate: {win_rate:.2f}")
        print(f"   - Avg Return: {avg_return:.1f}")

        ### Save best model
        if current_best_win_rate < win_rate:
            print("Saving New Best Model")
            print(f"   - Previous best win rate: {current_best_win_rate:.2f} -> {win_rate:.2f}")
            current_best_win_rate = win_rate

            save_path = config["save_path"]
            model.save(save_path)
        print("-"*60)

    total_time = (time.time() - start_time)
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f} seconds")


if __name__ == "__main__":
    train_env = SubprocVecEnv([make_env for _ in range(my_config["num_train_envs"])])

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    model = my_config["algorithm"](
        my_config["policy_network"],
        train_env,
        verbose=1,
        tensorboard_log=my_config["run_id"],
    )
    train(model, my_config)
