"""Rusher: the rule-based TA bot used as the medium baseline in eval_astro_party.py.

The same RuleBot as models/sniper_bot.py (read that file for how it plays), with a
different style: it fights up close, fires at looser aims, and doesn't dodge.
"""
from models.sniper_bot import RuleBot

rusher_bot_action = RuleBot(near=120.0, far=260.0, fire_tol=1.3, dodge=False)
