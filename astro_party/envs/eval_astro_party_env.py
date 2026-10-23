"""Grading environment: two players, both ships controlled from outside.
DON'T MODIFY: the TAs grade with the original version of this file.

eval_astro_party.py flies one ship with a TA bot (models/rusher_bot.py or
models/sniper_bot.py); the TAs' round robin flies both ships with students' models.
Same game as training (envs/game.py), but the reward is sparse: +1 to the
player who kills the enemy pilot (the final elimination that ends the duel),
-1 to the player killed, 0 otherwise. A draw at the step limit scores 0.
Don't train on this reward directly; shape your own in envs/train_astro_party_env.py.

    obs, infos = env.reset(seed=0)                       # obs = {"ship1": Box(84,), "ship2": Box(84,)}
    obs, rewards, terminations, truncations, infos = env.step({"ship1": a1, "ship2": a2})

Actions here are the game's native per-ship form, {"move": [turn, thrust],
"shoot": 0/1}; convert a model's Box(3,) output with envs.game.decode_action.
"""
import numpy as np
from gymnasium import spaces

from envs.game import FPS, SHIP_IDS, AstroPartyGame, observation_space
from envs.render import Renderer

KILL_REWARD = 1.0


class AstroPartyEvalEnv:
    agents = list(SHIP_IDS)
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode: str | None = None):
        self.game = AstroPartyGame()
        self.renderer = Renderer(render_mode)
        self.render_mode = render_mode
        act_space = spaces.Dict({"move": spaces.Box(-1, 1, (2,), dtype=np.float32), "shoot": spaces.Discrete(2)})
        self.observation_spaces = {sid: observation_space() for sid in self.agents}
        self.action_spaces = {sid: act_space for sid in self.agents}

    @property
    def ships(self):
        return self.game.ships

    def reset(self, seed=None, options=None):
        self.game.reset(seed)
        if self.render_mode == "human":
            self.render()
        return {sid: self.game.observe(sid) for sid in self.agents}, {sid: {} for sid in self.agents}

    def step(self, actions):
        events = self.game.step(actions)
        rewards = {sid: 0.0 for sid in self.agents}
        for e in events:
            if e.kind == "kill":
                rewards[e.by] += KILL_REWARD
                rewards[e.target] -= KILL_REWARD
        if self.render_mode == "human":
            self.render()
        obs = {sid: self.game.observe(sid) for sid in self.agents}
        terminations = {sid: self.game.done for sid in self.agents}
        truncations = {sid: self.game.truncated for sid in self.agents}
        infos = {sid: {} for sid in self.agents}
        return obs, rewards, terminations, truncations, infos

    def render(self):
        return self.renderer.draw(self.game)

    def close(self):
        self.renderer.close()
