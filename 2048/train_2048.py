import warnings
import time
import gymnasium as gym
from gymnasium.envs.registration import register

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import DQN

import numpy as np
import torch as th
from torch.nn import functional as F

# ---------------------------------------------------------------- Double DQN
# DoubleDQN subclasses SB3's DQN and only overrides train(), so a model trained
# with it is saved as an ordinary DQN checkpoint and eval_2048.py loads it with
# plain DQN.load(). Keep it that way: change only how the network learns (the TD
# target, loss, ...); don't override predict(), since eval_2048.py would not see it.
#   DoubleDQN(..., double_q=False)  -> plain DQN (identical to SB3's DQN)
#   DoubleDQN(..., double_q=True)   -> Double DQN, once you implement the TODO below
# Both log train/q_mean (the average Q-value the network predicts for the actions
# it took) to TensorBoard, so you can compare how DQN and Double DQN estimate values.
# References: SB3's DQN (https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/dqn.py)
# and the Double DQN paper (https://arxiv.org/abs/1509.06461).

class DoubleDQN(DQN):
    def __init__(self, *args, double_q: bool = True, **kwargs):
        self.double_q = double_q
        super().__init__(*args, **kwargs)

    def next_q_values(self, next_observations: th.Tensor) -> th.Tensor:
        """Value of the next state used in the TD target, shape (batch, 1)."""
        next_q_target = self.q_net_target(next_observations)  # (batch, n_actions), target network
        if not self.double_q:
            # DQN: the target network both chooses the best next action and evaluates it.
            next_q, _ = next_q_target.max(dim=1, keepdim=True)
            return next_q

        # TODO: Double DQN. Choose the best next action with the *online* network
        # (self.q_net), then evaluate that action with the *target* network
        # (next_q_target above). Return a (batch, 1) tensor.
        next_q_online = self.q_net(next_observations)
        online_max, online_idx = next_q_online.max(dim = 1, keepdim=True)
        next_q = next_q_target.gather(dim = 1, index = online_idx)
        return next_q



        


        raise NotImplementedError("Implement the Double DQN target in DoubleDQN.next_q_values()")

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Same as stable_baselines3's DQN.train(), except that the next-state value
        # comes from next_q_values() above, and the mean predicted Q is logged.
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses, q_means = [], []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                next_q = self.next_q_values(replay_data.next_observations)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())
            q_means.append(current_q_values.mean().item())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/q_mean", np.mean(q_means))


warnings.filterwarnings("ignore")
register(
    id='2048-v0',
    entry_point='envs:My2048Env'
)

# Set hyper params (configurations) for training
my_config = {
    "run_id": "example",
    # 2048 must be trained with DQN. DoubleDQN (above) with double_q=False is
    # exactly SB3's DQN plus Q-value logging; set double_q=True once you've
    # implemented the Double DQN target there, and compare the two.
    "algorithm": DoubleDQN,
    "algorithm_kwargs": {
        "double_q": True,
        "buffer_size": 100_000,  # SB3's default (1M) needs ~4 GB of RAM for 2048's observations
    },
    "policy_network": "MlpPolicy",
    "save_path": "models/model_2048",  # saved as models/model_2048.zip, the file you submit
    "num_train_envs": 4,
    "epoch_num": 50,
    "timesteps_per_epoch": 100,
    "eval_episode_num": 10,
}

def make_env():
    env = gym.make('2048-v0')
    return env

def eval(env, model, eval_episode_num):
    """Evaluate the model and return avg_score and avg_highest"""
    avg_score = 0
    avg_highest = 0
    for seed in range(eval_episode_num):
        done = False
        # Set seed using old Gym API
        env.seed(seed)
        obs = env.reset()

        # Interact with env using old Gym API
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        
        avg_highest += info[0]['highest']
        avg_score   += info[0]['score']

    avg_highest /= eval_episode_num
    avg_score /= eval_episode_num
        
    return avg_score, avg_highest

def train(eval_env, model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best_score = 0
    current_best_highest = 0

    start_time = time.time()

    for epoch in range(config["epoch_num"]):
        epoch_start_time = time.time()

        # Uncomment to enable wandb logging
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            # callback=WandbCallback(
            #     gradient_save_freq=100,
            #     verbose=2,
            # ),
        )

        epoch_duration = time.time() - epoch_start_time
        total_duration = time.time() - start_time

        ### Evaluation
        eval_start = time.time()
        avg_score, avg_highest = eval(eval_env, model, config["eval_episode_num"])
        eval_duration = time.time() - eval_start

        # Print training progress and speed
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['epoch_num']} completed")
        print(f"{'='*60}")
        print(f"Training Speed:")
        print(f"   - Epoch time: {epoch_duration:.1f}s")
        print(f"   - Eval time: {eval_duration:.1f}s")
        print(f"   - Total time: {total_duration/60:.1f} min")
        print(f"Performance:")
        print(f"   - Avg Score: {avg_score:.1f}")
        print(f"   - Avg Highest Tile: {avg_highest:.1f}")


        # wandb.log(
        #     {"avg_highest": avg_highest,
        #      "avg_score": avg_score}
        # )
        

        ### Save best model
        if current_best_score < avg_score or current_best_highest < avg_highest:
            print("Saving New Best Model")
            if current_best_score < avg_score:
                print(f"   - Previous best score: {current_best_score:.1f} → {avg_score:.1f}")
                current_best_score = avg_score
            if current_best_highest < avg_highest:
                print(f"   - Previous best tile: {current_best_highest:.1f} → {avg_highest:.1f}")
                current_best_highest = avg_highest

            save_path = config["save_path"]
            model.save(save_path)
        print("-"*60)
            
    total_time = (time.time() - start_time)
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f} seconds")

if __name__ == "__main__":

    # Create wandb session (Uncomment to enable wandb logging)
    # run = wandb.init(
    #     project="assignment_3",
    #     config=my_config,
    #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     id=my_config["run_id"]
    # )

    train_env = SubprocVecEnv([make_env for _ in range(my_config["num_train_envs"])])

    eval_env = DummyVecEnv([make_env])

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    model = my_config["algorithm"](
        my_config["policy_network"],
        train_env,
        verbose=1,
        tensorboard_log=my_config["run_id"],
        **my_config["algorithm_kwargs"],
    )
    train(eval_env, model, my_config)