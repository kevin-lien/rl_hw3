# rl_assignment_3

Two environments, both trained with Stable-Baselines3 (SB3):

- `2048/`: single-player 2048, trained with DQN (and the Double DQN you implement).
- `astro_party/`: a 1v1 top-down space duel. Destroying a ship ejects its pilot, and killing the pilot wins.

## Setup

1. Install Miniconda (a minimal conda): follow https://www.anaconda.com/docs/getting-started/miniconda/install
   for your OS, then open a new terminal so `conda` is on your PATH. Check with `conda --version`.
2. Create and activate one environment for the whole assignment (both games):

   ```bash
   conda create -n rl_assignment_3 python=3.10 -y
   conda activate rl_assignment_3
   ```
3. From the repo root, install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

Run `conda activate rl_assignment_3` in every new terminal. Run each game's scripts from inside its folder (`cd 2048` or `cd astro_party`).

## Files

```
2048/
  train_2048.py            training script, incl. the DoubleDQN template (edit freely)
  eval_2048.py             grading script (don't modify)
  envs/train_2048_env.py   training env (edit freely: termination, reward, ...)
  envs/eval_2048_env.py    grading env (don't modify)
  models/sample_model.zip  sample (dummy) checkpoint, loadable by eval_2048.py
astro_party/
  train_astro_party.py     training script; trains against rusher_bot by default (edit freely)
  eval_astro_party.py      grading script (don't modify)
  envs/train_astro_party_env.py   training env; the reward is the weights at the top and reward() (edit freely)
  envs/eval_astro_party_env.py    grading env (don't modify)
  envs/game.py             game logic, incl. the observation layout (don't modify)
  envs/render.py           graphics (don't modify)
  models/rusher_bot.py     rule-based TA bot (medium baseline)
  models/sniper_bot.py     rule-based TA bot (strong baseline); rusher uses the same code with a different style
  models/sample_model.zip  sample (dummy, untrained PPO) checkpoint, loadable by eval_astro_party.py
```

## 2048

Train with **DQN**: `train_2048.py` contains `DoubleDQN`, a subclass of SB3's `DQN`.

- `double_q=False` (the default) is exactly SB3's DQN. Train with it first.
- `double_q=True` switches to Double DQN once you implement the TODO in `DoubleDQN.next_q_values()`.
- Both log `train/q_mean`, the average Q-value the network predicts, to TensorBoard (`tensorboard --logdir <run_id>`). Compare the value curves of DQN and Double DQN.

You can submit either a DQN or a Double DQN checkpoint: both save as ordinary DQN checkpoints, so `eval_2048.py` loads them with plain `DQN.load()`. `train_2048.py` saves the best model as `models/model_2048.zip`.

```bash
cd 2048
python eval_2048.py --model models/model_2048.zip     # public seeds 0-99, prints your score out of 15
```

## Astro Party

- The training reward is entirely in `envs/train_astro_party_env.py`: a block of reward weights at the top and one `reward()` method. Grading uses `envs/eval_astro_party_env.py`, the same game with +1/−1 for the final kill.
- Every game starts from a random (seeded) spawn: your ship at a random safe spot, the opponent mirrored through the planet, never closer than 500 px. `AstroPartyTrainEnv(random_spawn=False)` uses fixed spawn points if you want an easier start, but grading always uses random spawns.
- The observation (`Box(84,)`) is documented entry by entry at the top of `envs/game.py`. The action is `Box(3,)` = `[turn, thrust, shoot_signal]`; `shoot_signal > 0` fires.
- You may duel against the TA bots to train your agent, and read or imitate their code.

```bash
cd astro_party
python eval_astro_party.py --model models/model_astro_party.zip            # public seeds 0-99, prints your score out of 10
python eval_astro_party.py --model models/model_astro_party.zip --render   # also watch the last duel against each bot
```

Your model plays with deterministic actions (`model.predict(obs, deterministic=True)`). A rollout counts as won only if your ship is the one left alive; a rollout still undecided after 1500 steps is a draw, and a draw is not a win. `train_astro_party.py` saves the best model as `models/model_astro_party.zip`.

## Grading

| Part | Points | How it's measured |
|---|---|---|
| 2048 simple baseline | 5 public + 5 private | reach 128 in at least one of 100 rollouts |
| 2048 medium baseline | 5 public + 5 private | reach 256 in at least one of 100 rollouts |
| 2048 strong baseline | 5 public + 5 private | reach 512 in at least one of 100 rollouts |
| Astro Party medium baseline (rusher_bot) | 5 public + 5 private | win rate ≥ 90% over 100 rollouts |
| Astro Party strong baseline (sniper_bot) | 5 public + 5 private | win rate ≥ 85% over 100 rollouts |
| Astro Party round robin | 20 (private) | rank among peers by match win rate; score decreases linearly with rank from 20 (1st) to 0 (last), ties share the better rank |
| Report | 30 | 2048 12 + Astro Party 18; see the assignment slides |

- **Round robin**: run privately by the TAs; any checkpoint `eval_astro_party.py` can load can enter. You play 10 games against each peer on TA-chosen seeds, and whoever wins more of the 10 games wins the match. Drawn games and tied matches are not wins.

- **Public test cases** use the default seeds of the eval scripts: 0–99 for both games. **Private test cases** use the same number of other seeds, chosen by the TAs (`--seed-base`). Public and private are tested separately.
- Each 2048 test case is limited to 10 minutes.
- Only one model per game is loaded and tested for all test cases. Models must be trained with SB3 algorithms, and we should be able to roughly reproduce your results.
- Parameter limits: at most **5 million** parameters for the 2048 model and **2 million** for the Astro Party model, excluding target networks.

## Submission

Submit one zip named after your student ID in lower case (e.g. `b09901171.zip`) on NTU COOL, containing a folder with that name:

```
b09901171/
  train_2048.py
  train_2048_env.py
  model_2048.zip
  train_astro_party.py
  train_astro_party_env.py
  model_astro_party.zip
  report.pdf
```

Leave out `__pycache__`, `.DS_Store`, etc. Deadline: 2026/11/04 (Wed) 11:59pm; no late submissions. A wrong file format costs 10 points; importing system-level packages such as `sys`, `os` or `shutil` scores 0. See the assignment slides for the full policy (plagiarism, LLM use).

Make sure the stock eval scripts can load your models:

- 2048 models must load with `DQN`; Astro Party models with one of PPO, A2C, SAC, TD3, DDPG.
- Use SB3's built-in policies, configured with `policy_kwargs` if you like (`net_arch`, `activation_fn`). A custom features extractor must be defined inside your training script; one imported from another file won't load on the grader.
- If you subclass an algorithm (like `DoubleDQN`), change only how it *learns*. The grader loads the base class, so a `predict()` override is silently dropped.
- The grader feeds your model the environment's raw observations. Observation wrappers and normalization such as `VecNormalize` are not saved inside the `.zip`.
- Before submitting, run the stock eval script on your `.zip` from a fresh copy of this repo.
